#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <boost/filesystem.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

namespace py = pybind11;

fast_gicp::NeighborSearchMethod search_method(const std::string& neighbor_search_method) {
  if(neighbor_search_method == "DIRECT1") {
    return fast_gicp::NeighborSearchMethod::DIRECT1;
  } else if (neighbor_search_method == "DIRECT7") {
    return fast_gicp::NeighborSearchMethod::DIRECT7;
  } else if (neighbor_search_method == "DIRECT27") {
    return fast_gicp::NeighborSearchMethod::DIRECT27;
  } else if (neighbor_search_method == "DIRECT_RADIUS") {
    return fast_gicp::NeighborSearchMethod::DIRECT_RADIUS;
  }

  std::cerr << "error: unknown neighbor search method " << neighbor_search_method << std::endl;
  return fast_gicp::NeighborSearchMethod::DIRECT1;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr eigen2pcl(const Eigen::Matrix<double, -1, 3>& points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->resize(points.rows());

  for(int i=0; i<points.rows(); i++) {
    cloud->at(i).getVector3fMap() = points.row(i).cast<float>();
  }
  return cloud;
}

Eigen::Matrix<double, -1, 3> downsample(const Eigen::Matrix<double, -1, 3>& points, double downsample_resolution) {
  auto cloud = eigen2pcl(points);

  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
  voxelgrid.setInputCloud(cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  voxelgrid.filter(*filtered);

  Eigen::Matrix<float, -1, 3> filtered_points(filtered->size(), 3);
  for(int i=0; i<filtered->size(); i++) {
    filtered_points.row(i) = filtered->at(i).getVector3fMap();
  }

  return filtered_points.cast<double>();
}

Eigen::Matrix4d align_points(
  const Eigen::Matrix<double, -1, 3>& target,
  const Eigen::Matrix<double, -1, 3>& source,
  const std::string& method,
  double downsample_resolution,
  int k_correspondences,
  double max_correspondence_distance,
  double voxel_resolution,
  int num_threads,
  const std::string& neighbor_search_method,
  double neighbor_search_radius,
  const Eigen::Matrix4f& initial_guess
) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = eigen2pcl(target);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = eigen2pcl(source);

  if(downsample_resolution > 0.0) {
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid.setInputCloud(target_cloud);
    voxelgrid.filter(*filtered);
    target_cloud.swap(filtered);

    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*filtered);
    source_cloud.swap(filtered);
  }
  std::shared_ptr<fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>> reg;
  if(method == "GICP") {
    std::shared_ptr<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>> gicp(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>);
    gicp->setMaxCorrespondenceDistance(max_correspondence_distance);
    gicp->setCorrespondenceRandomness(k_correspondences);
    gicp->setNumThreads(num_threads);
    reg = gicp;
  } else if (method == "VGICP") {
    std::shared_ptr<fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>> vgicp(new fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>);
    vgicp->setCorrespondenceRandomness(k_correspondences);
    vgicp->setResolution(voxel_resolution);
    vgicp->setNeighborSearchMethod(search_method(neighbor_search_method));
    vgicp->setNumThreads(num_threads);
    reg = vgicp;
  } else if (method == "VGICP_CUDA") {
#ifdef USE_VGICP_CUDA
    std::shared_ptr<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>> vgicp(new fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>);
    vgicp->setCorrespondenceRandomness(k_correspondences);
    vgicp->setNeighborSearchMethod(search_method(neighbor_search_method), neighbor_search_radius);
    vgicp->setResolution(voxel_resolution);
    reg = vgicp;
#else
    std::cerr << "error: you need to build fast_gicp with BUILD_VGICP_CUDA=ON" << std::endl;
    return Eigen::Matrix4d::Identity();
#endif
  } else if (method == "NDT_CUDA") {
#ifdef USE_VGICP_CUDA
    std::shared_ptr<fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>> ndt(new fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>);
    ndt->setResolution(voxel_resolution);
    ndt->setNeighborSearchMethod(search_method(neighbor_search_method), neighbor_search_radius);
    reg = ndt;
#else
    std::cerr << "error: you need to build fast_gicp with BUILD_VGICP_CUDA=ON" << std::endl;
    return Eigen::Matrix4d::Identity();
#endif
  } else {
    std::cerr << "error: unknown registration method " << method << std::endl;
    return Eigen::Matrix4d::Identity();
  }
  reg->setInputTarget(target_cloud);
  reg->setInputSource(source_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
  reg->align(*aligned, initial_guess);
  return reg->getFinalTransformation().cast<double>();
}


using LsqRegistration = fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
using FastGICP = fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>;
using FastVGICP = fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
#ifdef USE_VGICP_CUDA
using FastVGICPCuda = fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>;
using NDTCuda = fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>;
#endif

PYBIND11_MODULE(pygicp, m) {
  m.def("downsample", &downsample, "downsample points");

  m.def("align_points", &align_points, "align two point sets",
    py::arg("target"),
    py::arg("source"),
    py::arg("method") = "GICP",
    py::arg("downsample_resolution") = -1.0,
    py::arg("k_correspondences") = 15,
    py::arg("max_correspondence_distance") = std::numeric_limits<double>::max(),
    py::arg("voxel_resolution") = 1.0,
    py::arg("num_threads") = 0,
    py::arg("neighbor_search_method") = "DIRECT1",
    py::arg("neighbor_search_radius") = 1.5,
    py::arg("initial_guess") = Eigen::Matrix4f::Identity()
  );

  py::class_<LsqRegistration, std::shared_ptr<LsqRegistration>>(m, "LsqRegistration")
    .def("set_input_target", [] (LsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputTarget(eigen2pcl(points)); })
    .def("set_input_source", [] (LsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputSource(eigen2pcl(points)); })
    .def("swap_source_and_target", &LsqRegistration::swapSourceAndTarget)
    .def("get_final_hessian", &LsqRegistration::getFinalHessian)
    .def("get_final_transformation", &LsqRegistration::getFinalTransformation)
    .def("get_fitness_score", [] (LsqRegistration& reg, const double max_range) { return reg.getFitnessScore(max_range); })
    .def("align",
      [] (LsqRegistration& reg, const Eigen::Matrix4f& initial_guess) { 
        pcl::PointCloud<pcl::PointXYZ> aligned;
        reg.align(aligned, initial_guess);
        return reg.getFinalTransformation();
      }, py::arg("initial_guess") = Eigen::Matrix4f::Identity()
    )
  ;
  py::class_<FastGICP, LsqRegistration, std::shared_ptr<FastGICP>>(m, "FastGICP")
    .def(py::init())
    .def(py::pickle(
        [](const FastGICP &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getSourceRotationsq());
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            // FastGICP p(t[0].cast<std::string>());
            FastGICP p;

            /* Assign any additional state */
            // p.setExtra(t[1].cast<int>());

            return p;
        }
    ))
    .def("set_num_threads", &FastGICP::setNumThreads)
    .def("set_correspondence_randomness", &FastGICP::setCorrespondenceRandomness)
    .def("set_max_correspondence_distance", &FastGICP::setMaxCorrespondenceDistance)
    .def("set_max_knn_distance", &FastGICP::setKNNMaxDistance)
    .def("get_source_rotationsq", [] (FastGICP& gicp){
      return py::array(gicp.getSourceRotationsqSize(), gicp.getSourceRotationsq().data());
      // std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> source_rotationsq = gicp.getSourceRotationsq();
      // py::list data; for(const auto& q:source_rotationsq) data.append(q); return data;
      })
    .def("get_target_rotationsq", [] (FastGICP& gicp){
      return py::array(gicp.getTargetRotationsqSize(), gicp.getTargetRotationsq().data());
      // std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> target_rotationsq = gicp.getTargetRotationsq();
      // py::list data; for(const auto& q:target_rotationsq) data.append(q); return data;
      })
    .def("get_source_scales", [] (FastGICP& gicp){
      return py::array(gicp.getSourceScaleSize(), gicp.getSourceScales().data());
      // std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> source_scales = gicp.getSourceScales();
      // py::list data; for(const auto& s:source_scales) data.append(s); return data;
      })
    .def("get_target_scales", [] (FastGICP& gicp){
      return py::array(gicp.getTargetScaleSize(), gicp.getTargetScales().data());
      // std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> target_scales = gicp.getTargetScales();
      // py::list data; for(const auto& s:target_scales) data.append(s); return data;
      })
    .def("calculate_source_covariance", &FastGICP::calculateSourceCovariance)
    .def("calculate_target_covariance", &FastGICP::calculateTargetCovariance)
    .def("calculate_target_covariance_withz", &FastGICP::calculateTargetCovarianceWithZ)
    .def("calculate_target_covariance_with_filter", &FastGICP::calculateTargetCovarianceWithFilter)
    .def("get_source_correspondence", [] (FastGICP& gicp){
    	return py::make_tuple(py::array(gicp.getSourceSize(), gicp.getSourceCorrespondences().data()), 
    				py::array(gicp.getSourceSize(), gicp.getSourceSqDistances().data()));
    })
    .def("set_source_covariances_fromqs", [] (FastGICP& gicp, py::array rotationsq, py::array scales){
    	if(py::len(rotationsq)/4!=py::len(scales)/3){ std::cerr<<"qs size not matched" <<std::endl; return;}
    	const auto input_rotationsq = rotationsq.cast<std::vector<float>>();
    	const auto input_scales = scales.cast<std::vector<float>>();
    	gicp.setSourceCovariances(input_rotationsq, input_scales);
    })
    .def("set_target_covariances_fromqs", [] (FastGICP& gicp, py::array rotationsq, py::array scales){
    	if(py::len(rotationsq)/4!=py::len(scales)/3){ std::cerr<<"qs size not matched" <<std::endl; return;}
    	const auto input_rotationsq = rotationsq.cast<std::vector<float>>();
    	const auto input_scales = scales.cast<std::vector<float>>();
    	gicp.setTargetCovariances(input_rotationsq, input_scales);
    })
    .def("set_source_z_values", [] (FastGICP& gicp, py::array z_values){
      const auto input_z_values = z_values.cast<std::vector<float>>();
    	gicp.setSourceZvalues(input_z_values);
    })
    .def("set_target_z_values", [] (FastGICP& gicp, py::array z_values){
      const auto input_z_values = z_values.cast<std::vector<float>>();
    	gicp.setTargetZvalues(input_z_values);
    })
    .def("set_source_filter", [] (FastGICP& gicp, py::int_ num_trackable, py::array filter){
      const auto input_filter = filter.cast<std::vector<int>>();
    	gicp.setSourceFilter(num_trackable, input_filter);
    })
    .def("set_target_filter", [] (FastGICP& gicp, py::int_ num_trackable, py::array filter){
      const auto input_filter = filter.cast<std::vector<int>>();
    	gicp.setTargetFilter(num_trackable, input_filter);
    })
  ;

  py::class_<FastVGICP, FastGICP, std::shared_ptr<FastVGICP>>(m, "FastVGICP")
    .def(py::init())
    .def("set_resolution", &FastVGICP::setResolution)
    .def("set_neighbor_search_method", [](FastVGICP& vgicp, const std::string& method) { vgicp.setNeighborSearchMethod(search_method(method)); })
    .def("get_voxel_mean_cov", [](FastVGICP& vgicp){ 
      using meanvec = typename std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>;
      using covvec = typename std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;
      std::pair<meanvec, covvec> data = vgicp.getVoxelMeanCov();
//      return py::make_tuple(data.first, data.second);      
      py::list data0; py::list data1;
      for (const auto& d:data.first) data0.append(d);
      for (const auto& d:data.second) data1.append(d);
      return py::make_tuple(data0, data1);
    })
  ;

#ifdef USE_VGICP_CUDA
  py::class_<FastVGICPCuda, LsqRegistration, std::shared_ptr<FastVGICPCuda>>(m, "FastVGICPCuda")
    .def(py::init())
    .def("set_resolution", &FastVGICPCuda::setResolution)
    .def("set_neighbor_search_method",
      [](FastVGICPCuda& vgicp, const std::string& method, double radius) { vgicp.setNeighborSearchMethod(search_method(method), radius); }
      , py::arg("method") = "DIRECT1", py::arg("radius") = 1.5
    )
    .def("set_correspondence_randomness", &FastVGICPCuda::setCorrespondenceRandomness)
  ;

  py::class_<NDTCuda, LsqRegistration, std::shared_ptr<NDTCuda>>(m, "NDTCuda")
    .def(py::init())
    .def("set_neighbor_search_method",
      [](NDTCuda& ndt, const std::string& method, double radius) { ndt.setNeighborSearchMethod(search_method(method), radius); }
      , py::arg("method") = "DIRECT1", py::arg("radius") = 1.5
    )
    .def("set_resolution", &NDTCuda::setResolution)
  ;
#endif

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

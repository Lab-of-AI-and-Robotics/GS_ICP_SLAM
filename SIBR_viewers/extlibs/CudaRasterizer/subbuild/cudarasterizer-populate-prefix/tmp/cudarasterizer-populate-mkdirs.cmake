# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/CudaRasterizer"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/build"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/subbuild/cudarasterizer-populate-prefix"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/subbuild/cudarasterizer-populate-prefix/tmp"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/subbuild/cudarasterizer-populate-prefix/src/cudarasterizer-populate-stamp"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/subbuild/cudarasterizer-populate-prefix/src"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/subbuild/cudarasterizer-populate-prefix/src/cudarasterizer-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/subbuild/cudarasterizer-populate-prefix/src/cudarasterizer-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/CudaRasterizer/subbuild/cudarasterizer-populate-prefix/src/cudarasterizer-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

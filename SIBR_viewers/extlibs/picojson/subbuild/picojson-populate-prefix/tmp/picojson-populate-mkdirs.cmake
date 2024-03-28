# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/picojson"
  "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/build"
  "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/subbuild/picojson-populate-prefix"
  "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/subbuild/picojson-populate-prefix/tmp"
  "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/subbuild/picojson-populate-prefix/src/picojson-populate-stamp"
  "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/subbuild/picojson-populate-prefix/src"
  "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/subbuild/picojson-populate-prefix/src/picojson-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/subbuild/picojson-populate-prefix/src/picojson-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/lair99/gaussian_proj/RGBD_gaussian-splatting/SIBR_viewers/extlibs/picojson/subbuild/picojson-populate-prefix/src/picojson-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

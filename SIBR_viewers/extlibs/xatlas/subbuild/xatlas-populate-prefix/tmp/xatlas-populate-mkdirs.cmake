# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/xatlas"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/build"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/subbuild/xatlas-populate-prefix"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/subbuild/xatlas-populate-prefix/tmp"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/subbuild/xatlas-populate-prefix/src/xatlas-populate-stamp"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/subbuild/xatlas-populate-prefix/src"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/subbuild/xatlas-populate-prefix/src/xatlas-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/subbuild/xatlas-populate-prefix/src/xatlas-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/subbuild/xatlas-populate-prefix/src/xatlas-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

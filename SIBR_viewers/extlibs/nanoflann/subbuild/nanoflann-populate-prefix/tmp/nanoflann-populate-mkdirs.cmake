# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/nanoflann"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/build"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/tmp"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

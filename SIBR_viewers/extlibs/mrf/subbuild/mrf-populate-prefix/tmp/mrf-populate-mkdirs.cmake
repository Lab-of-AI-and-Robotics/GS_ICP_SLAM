# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/mrf"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/build"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/subbuild/mrf-populate-prefix"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/subbuild/mrf-populate-prefix/tmp"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/subbuild/mrf-populate-prefix/src/mrf-populate-stamp"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/subbuild/mrf-populate-prefix/src"
  "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/subbuild/mrf-populate-prefix/src/mrf-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/subbuild/mrf-populate-prefix/src/mrf-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/subbuild/mrf-populate-prefix/src/mrf-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

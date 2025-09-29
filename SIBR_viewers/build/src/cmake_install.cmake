# Install script for directory: /home/GS_ICP_SLAM/SIBR_viewers/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/GS_ICP_SLAM/SIBR_viewers/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/extlibs/imgui/build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "imgui_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libimgui.a")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/lib" TYPE STATIC_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/imgui/build/libimgui.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "imgui_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libimgui.a")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/bin" TYPE STATIC_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/imgui/build/libimgui.a")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nativefiledialog/build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "nativefiledialog_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libnativefiledialog.a")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/lib" TYPE STATIC_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nativefiledialog/build/libnativefiledialog.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "nativefiledialog_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libnativefiledialog.a")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/bin" TYPE STATIC_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nativefiledialog/build/libnativefiledialog.a")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mrf_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libmrf.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/lib" TYPE SHARED_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/build/libmrf.so")
    if(EXISTS "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libmrf.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libmrf.so")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libmrf.so")
      endif()
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mrf_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mrf_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libmrf.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/bin" TYPE SHARED_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/mrf/build/libmrf.so")
    if(EXISTS "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libmrf.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libmrf.so")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libmrf.so")
      endif()
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mrf_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/extlibs/nanoflann/build/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/extlibs/picojson/build/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/extlibs/rapidxml/build/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "xatlas_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libxatlas.so.1.0")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/lib" TYPE SHARED_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/build/libxatlas.so.1.0")
    if(EXISTS "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libxatlas.so.1.0" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libxatlas.so.1.0")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libxatlas.so.1.0")
      endif()
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "xatlas_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libxatlas.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/lib" TYPE SHARED_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/build/libxatlas.so")
    if(EXISTS "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libxatlas.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libxatlas.so")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libxatlas.so")
      endif()
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "xatlas_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libxatlas.so.1.0")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/bin" TYPE SHARED_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/build/libxatlas.so.1.0")
    if(EXISTS "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libxatlas.so.1.0" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libxatlas.so.1.0")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libxatlas.so.1.0")
      endif()
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "xatlas_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libxatlas.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/bin" TYPE SHARED_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/extlibs/xatlas/build/libxatlas.so")
    if(EXISTS "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libxatlas.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libxatlas.so")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libxatlas.so")
      endif()
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/system/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/graphics/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/renderer/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/raycaster/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/view/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/scene/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/assets/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/imgproc/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/video/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/projects/basic/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/projects/gaussianviewer/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/GS_ICP_SLAM/SIBR_viewers/build/src/projects/remote/cmake_install.cmake")
endif()


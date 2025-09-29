# Install script for directory: /home/GS_ICP_SLAM/SIBR_viewers/src/core/view

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_colored_mesh.frag;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_colored_mesh.vert;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_colored_per_triangle_normals.geom;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_colored_per_triangle_normals.vert;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_colored_per_vertex_normals.geom;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_colored_per_vertex_normals.vert;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_points.frag;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_points.vert;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_uv_tex.frag;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alpha_uv_tex_array.frag;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alphaimgview.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/alphaimgview.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/anaglyph.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/anaglyph.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/axisgizmo.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/axisgizmo.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/camstub.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/camstub.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/depth.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/depth.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/depthonly.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/depthonly.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/image_viewer.frag;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/image_viewer.vert;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/mesh_color.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/mesh_color.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/mesh_debugview.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/mesh_debugview.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/mesh_normal.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/mesh_normal.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/number.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/number.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/skybox.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/skybox.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/text-imgui.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/text-imgui.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/texture.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/texture.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/topview.fp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/topview.vp;/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core/uv_mesh.vert")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/shaders/core" TYPE PROGRAM FILES
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_colored_mesh.frag"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_colored_mesh.vert"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_colored_per_triangle_normals.geom"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_colored_per_triangle_normals.vert"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_colored_per_vertex_normals.geom"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_colored_per_vertex_normals.vert"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_points.frag"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_points.vert"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_uv_tex.frag"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alpha_uv_tex_array.frag"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alphaimgview.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/alphaimgview.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/anaglyph.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/anaglyph.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/axisgizmo.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/axisgizmo.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/camstub.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/camstub.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/depth.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/depth.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/depthonly.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/depthonly.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/image_viewer.frag"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/image_viewer.vert"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/mesh_color.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/mesh_color.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/mesh_debugview.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/mesh_debugview.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/mesh_normal.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/mesh_normal.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/number.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/number.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/skybox.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/skybox.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/text-imgui.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/text-imgui.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/texture.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/texture.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/topview.fp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/topview.vp"
      "/home/GS_ICP_SLAM/SIBR_viewers/src/core/view/shaders/uv_mesh.vert"
      )
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libsibr_view.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/lib" TYPE SHARED_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/view/libsibr_view.so")
    if(EXISTS "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libsibr_view.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libsibr_view.so")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/lib/libsibr_view.so")
      endif()
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libsibr_view.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/home/GS_ICP_SLAM/SIBR_viewers/install/bin" TYPE SHARED_LIBRARY FILES "/home/GS_ICP_SLAM/SIBR_viewers/build/src/core/view/libsibr_view.so")
    if(EXISTS "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libsibr_view.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libsibr_view.so")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/GS_ICP_SLAM/SIBR_viewers/install/bin/libsibr_view.so")
      endif()
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  endif()
endif()


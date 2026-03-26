cmake_minimum_required(VERSION 3.16)
project(assembly_mesh_repair_cgal_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

set(_VENDORED_CGAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/CGAL-6.1.1")
if(EXISTS "${_VENDORED_CGAL_DIR}/CGALConfig.cmake")
  list(PREPEND CMAKE_PREFIX_PATH "${_VENDORED_CGAL_DIR}")
  message(STATUS "Trying vendored CGAL from ${_VENDORED_CGAL_DIR}")
endif()

find_package(CGAL QUIET CONFIG)
if(NOT CGAL_FOUND)
  message(FATAL_ERROR "CGAL was not found. Install libcgal-dev (and GMP/MPFR development packages) or place CGAL 6.1.1 under third_party/CGAL-6.1.1.")
endif()

if(DEFINED CGAL_VERSION)
  message(STATUS "Using CGAL ${CGAL_VERSION}")
endif()

add_executable(check_self_intersections check_self_intersections.cpp)
target_link_libraries(check_self_intersections PRIVATE CGAL::CGAL)
target_include_directories(check_self_intersections PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(autorefine_obj autorefine_obj.cpp)
target_link_libraries(autorefine_obj PRIVATE CGAL::CGAL)
target_include_directories(autorefine_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

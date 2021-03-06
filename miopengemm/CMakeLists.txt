################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic " )

set( miopengemm_INSTALL_DIR miopengemm)

option(BUILD_DEV OFF)

if(BUILD_DEV)
configure_file(config.hpp.dev dev_include/miopengemm/config.hpp)
else()
configure_file(config.hpp.install dev_include/miopengemm/config.hpp)
endif()

file(GLOB_RECURSE source_files src/*.cpp)

add_library(miopengemm ${source_files})

target_link_libraries(miopengemm ${OPENCL_LIBRARIES})

target_include_directories (miopengemm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/dev_include>
    ${OPENCL_INCLUDE_DIRS})

rocm_install_targets(
  TARGETS miopengemm
  INCLUDE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PREFIX ${miopengemm_INSTALL_DIR}
)

rocm_export_targets(
  TARGETS miopengemm
  PREFIX ${miopengemm_INSTALL_DIR}
)

rocm_create_package(
    NAME miopengemm
    DESCRIPTION "A tool for generating OpenCL matrix multiplication (GEMM) kernels"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
)

rocm_install_symlink_subdir(${miopengemm_INSTALL_DIR})


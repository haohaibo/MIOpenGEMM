SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic " )

include(InstallSymlinks)
include(GNUInstallDirs)
set( miopengemm_INSTALL_DIR miopengemm)
set( BIN_INSTALL_DIR ${miopengemm_INSTALL_DIR}/${CMAKE_INSTALL_BINDIR} )
set( LIB_INSTALL_DIR ${miopengemm_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${miopengemm_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR} )
set( DATA_INSTALL_DIR ${miopengemm_INSTALL_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/miopengemm )
set( CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/miopengemm )

include(CreatePackage)
create_package(
    NAME miopengemm
    DESCRIPTION "A tool for generating OpenCL matrix multiplication (GEMM) kernels"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG ${LIB_INSTALL_DIR}
)

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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/dev_include>
    ${OPENCL_INCLUDE_DIRS})

target_include_directories (miopengemm SYSTEM INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)

############################################################
# The following is cmake code to generate a config file package for Tensile
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( PackageConfigHelpers )



configure_package_config_file(
  miopengemm-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/miopengemm-config.cmake
  INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
  PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR CONFIG_PACKAGE_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/miopengemm-config-version.cmake
  VERSION ${miopengemm_VERSION}
  COMPATIBILITY SameMajorVersion
)

############################################################
# Library install logic

install(TARGETS miopengemm 
    EXPORT miopengemm-targets
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR})

install(DIRECTORY include/miopengemm DESTINATION ${INCLUDE_INSTALL_DIR})

install( EXPORT miopengemm-targets
  DESTINATION
    ${CONFIG_PACKAGE_INSTALL_DIR}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/miopengemm-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/miopengemm-config-version.cmake
  DESTINATION
    ${CONFIG_PACKAGE_INSTALL_DIR} )

install(DIRECTORY
  python
  opencl
  DESTINATION ${DATA_INSTALL_DIR}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.cl"
)

install_symlink_subdir(${miopengemm_INSTALL_DIR})

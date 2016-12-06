cmake_minimum_required(VERSION 3.6)
project(doxygen_builder C CXX)
include(ExternalProject)

Project_Add(doxygen
        GIT_REPOSITORY https://github.com/doxygen/doxygen.git
        CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
        -DCMAKE_CXX_FLAGS=${MSVC_COMPILER_DEFS}
        -Dgtest_force_shared_crt=${GTEST_FORCE_SHARED_CRT}
        -Dgtest_disable_pthreads=${GTEST_DISABLE_PTHREADS}
        -DBUILD_GTEST=ON
        PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
        # Disable install step
        INSTALL_COMMAND ""
        )

# Specify include dir
ExternalProject_Get_Property(doxygen source_dir)
set(DOXY_INCLUDE_DIRS ${source_dir}/doxygen/include PARENT_SCOPE)

# Specify link libraries
ExternalProject_Get_Property(doxygen binary_dir)
set(DOXY_LIBS_DIR ${binary_dir}/doxygen PARENT_SCOPE)
cmake_minimum_required(VERSION 3.6)
project(gtest_builder C CXX)
include(ExternalProject)

set(GTEST_FORCE_SHARED_CRT ON)
set(GTEST_DISABLE_PTHREADS OFF)

if(MINGW)
    set(GTEST_DISABLE_PTHREADS ON)
endif()

ExternalProject_Add(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
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
ExternalProject_Get_Property(googlebenchmark source_dir)
set(GBENCH_INCLUDE_DIRS ${source_dir}/googlebenchmark/include PARENT_SCOPE)

# Specify MainTest's link libraries
ExternalProject_Get_Property(googlebenchmark binary_dir)
set(GBENCH_LIBS_DIR ${binary_dir}/googlebenchmark/gtest PARENT_SCOPE)
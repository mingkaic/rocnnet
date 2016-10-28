cmake_minimum_required(VERSION 3.6)
set(PROJECT_NAME_STR cnnet)
project(${PROJECT_NAME_STR} C CXX)

find_package(Threads REQUIRED)

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#-------------------
# Other flags
#-------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -ansi -Wno-deprecated -pthread)
endif()

if (MSVC)
    #vc 2012 fix for vararg templates
    set(MSVC_COMPILER_DEFS "-D_VARIADIC_MAX=10")
endif()

#-------------------
# Build TENNCOR Library
#-------------------
FILE(GLOB TenncorSources
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/*/*.cpp)
set(PROJECT_LIB_NAME tenncor)
set(TENNCOR_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${TENNCOR_INCLUDE_DIR})
add_library(${PROJECT_LIB_NAME} STATIC ${TenncorSources})
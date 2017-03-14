cmake_minimum_required(VERSION 2.8)
project(tenncor C CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
option(RunTest "Test" OFF)
option(EdgeVis "Visualize Graph Edges" OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wno-reorder -Wall -ansi -Wno-deprecated -pthread)
endif()

if (MSVC)
    #vc 2012 fix for vararg templates
    set(MSVC_COMPILER_DEFS "-D_VARIADIC_MAX=10")
endif()

#>>>>>>>>>>>>>>>>>>>> TENNCOR SPECIFIC <<<<<<<<<<<<<<<<<<<<
# TODO: make option
if (EdgeVis)
    ADD_DEFINITIONS(-DEDGE_RCD="op-profile.csv")
endif()

set(TENNCOR_INCLUDE include)
FILE(GLOB_RECURSE TENNCOR_SRC src/*.cpp)

# set tenncor include path
include_directories(${TENNCOR_INCLUDE})

set(TENNCOR_LIB tenncor)
add_library(${TENNCOR_LIB} STATIC ${TENNCOR_SRC})

#>>>>>>>>>>>>>>>>>>>> GOOGLE TEST <<<<<<<<<<<<<<<<<<<<
if (RunTest)
    # todo: add instrumentation here
    set(TENNCOR_LIB_TEST tenncor-inst)
    add_library(${TENNCOR_LIB_TEST} STATIC ${TENNCOR_SRC}
                libsrc/graph.cpp libsrc/tensor.cpp)

    add_subdirectory(ext/gtest)
    add_subdirectory(tests)
endif()

cmake_minimum_required(VERSION 2.8)
project(tenncor C CXX)

find_package(Threads REQUIRED)

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on) 

option(TENNCOR_TEST "Test" OFF)
option(VIS_EDGE "Visualize Graph Edges" OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-Wno-reorder -Wall -ansi -Wno-deprecated -pthread)
endif()

if (MSVC)
	#vc 2012 fix for vararg templates
	set(MSVC_COMPILER_DEFS "-D_VARIADIC_MAX=10")
endif()

#>>>>>>>>>>>>>>>>>>>> TENNCOR SPECIFIC <<<<<<<<<<<<<<<<<<<<
if (VIS_EDGE)
	# visualize tenncor edges
	ADD_DEFINITIONS(-DEDGE_RCD="op-profile.csv")
endif(VIS_EDGE)

set(TENNCOR_INCLUDE include)
FILE(GLOB_RECURSE TENNCOR_SRC src/*.cpp)

# set tenncor include path
include_directories(${TENNCOR_INCLUDE})

set(TENNCOR_LIB tenncor)
add_library(${TENNCOR_LIB} STATIC ${TENNCOR_SRC})

#>>>>>>>>>>>>>>>>>>>> GOOGLE TEST <<<<<<<<<<<<<<<<<<<<
if (TENNCOR_TEST)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_TENSOR_MODULE_TESTS -DDISABLE_MEMORY_MODULE_TESTS")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_REACT_TEST -DDISABLE_NODE_TEST -DDISABLE_CONNECTOR_TEST") #todo: remove to enable tests
	# for testing
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") # debug, no optimisation
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
	set(TENNCOR_LIB_TEST tenncor-inst)
	add_library(${TENNCOR_LIB_TEST} STATIC ${TENNCOR_SRC}
		declare/graph.cpp declare/tensor.cpp)

	add_subdirectory(ext/gtest)
	add_subdirectory(tests)
endif(TENNCOR_TEST)

cmake_minimum_required(VERSION 2.8)
project(tenncor C CXX)
set(TENNCOR_LIB tenncor)

find_package(Threads REQUIRED)

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(TENNCOR_TEST "Test" OFF)
option(VIS_EDGE "Visualize Graph Edges" OFF)
option(LCOVERAGE "Evaluate Coverage" ON)

# enable modern c++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# enable warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

#>>>>>>>>>>>>>>>>>>>> TENNCOR SPECIFIC <<<<<<<<<<<<<<<<<<<<
if (VIS_EDGE)
	# visualize tenncor edges
	ADD_DEFINITIONS(-DEDGE_RCD="op-profile.csv")
endif(VIS_EDGE)

include(FindProtobuf)
add_subdirectory(proto)

FILE(GLOB_RECURSE TENNCOR_SRC src/*.cpp)

# set tenncor include path
set(TENNCOR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TENNCOR_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR})
add_library(${TENNCOR_LIB} STATIC ${TENNCOR_SRC})
target_link_libraries(${TENNCOR_LIB} proto ${PROTOBUF_LIBRARY})

#>>>>>>>>>>>>>>>>>>>> GOOGLE TEST <<<<<<<<<<<<<<<<<<<<
if (TENNCOR_TEST)
	#todo: remove to enable tests
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_TENSOR_MODULE_TESTS -DDISABLE_MEMORY_MODULE_TESTS")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_REACT_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_NODE_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_CONNECTOR_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_IMMUTABLE_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_LEAF_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_CONSTANT_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_IVARIABLE_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_VARIABLE_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_PLACEHOLDER_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_ELEMENTARY_TEST")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_TRANSFORM_TEST")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_MATMUL_TEST")
	# coverage
	if (LCOVERAGE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") # debug, no optimisation
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
	endif(LCOVERAGE)
	set(TENNCOR_LIB_TEST tenncor-inst)
	add_library(${TENNCOR_LIB_TEST} STATIC ${TENNCOR_SRC}
		declare/graph.cpp declare/tensor.cpp)
	target_link_libraries(${TENNCOR_LIB_TEST} proto ${PROTOBUF_LIBRARY})

	add_subdirectory(ext/gtest)
	add_subdirectory(tests)
endif(TENNCOR_TEST)

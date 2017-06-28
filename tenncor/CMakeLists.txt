cmake_minimum_required(VERSION 2.8)
project(tenncor C CXX)

if (NOT TENNCOR_LIB)
	set(TENNCOR_LIB tenncor)
endif()

if (NOT TENNCOR_INCLUDE)
	set(TENNCOR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Threads REQUIRED)

option(TENNCOR_TEST "Test" OFF)
option(VIS_EDGE "Visualize Graph Edges" OFF)
option(LCOVERAGE "Evaluate Coverage" ON)

# enable modern c++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# enable warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

#>>>>>>>>>>>>>>>>>>>> TENNCOR SPECIFIC <<<<<<<<<<<<<<<<<<<<
include(FindProtobuf)
add_subdirectory(proto)

FILE(GLOB_RECURSE TENNCOR_SRC src/*.cpp)
FILE(GLOB_RECURSE DB_DECLARE declare/*.cpp)

#>>>>>>>>>>>>>>>>>>>> MAKE TENNCOR LIB <<<<<<<<<<<<<<<<<<<<
if (VIS_EDGE)
	# visualize tenncor edges
	message("ALLOWING VISUALIZATION OF GRAPH: WARNING (additional overhead)")
	add_definitions(-DEDGE_RCD="op-profile.csv")
endif()

# static version
include_directories(${TENNCOR_INCLUDE})
add_library(${TENNCOR_LIB} STATIC ${TENNCOR_SRC})
target_link_libraries(${TENNCOR_LIB} proto ${PROTOBUF_LIBRARY})

#>>>>>>>>>>>>>>>>>>>> GOOGLE TEST <<<<<<<<<<<<<<<<<<<<
if (TENNCOR_TEST)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_TENSOR_MODULE_TESTS")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_MEMORY_MODULE_TESTS")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_TOP_NODE_MODULE_TESTS")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_LEAF_MODULE_TESTS")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_GRAPH_MODULE_TESTS")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_MERGE_IMM_TEST")

	# coverage
	if (LCOVERAGE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") # debug, no optimisation
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
	endif(LCOVERAGE)
	set(TENNCOR_LIB_TEST tenncor-inst)
	add_library(${TENNCOR_LIB_TEST} STATIC ${TENNCOR_SRC} ${DB_DECLARE})
	target_link_libraries(${TENNCOR_LIB_TEST} proto ${PROTOBUF_LIBRARY})

	add_subdirectory(ext/gtest)
	add_subdirectory(tests)
endif()

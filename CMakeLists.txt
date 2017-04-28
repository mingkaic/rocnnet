cmake_minimum_required(VERSION 2.8)
project(rocnnet C CXX)

find_package(Threads REQUIRED)
option(LCOVERAGE "Evaluate Coverage" ON)

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(TENNCOR_DIR ${PROJECT_SOURCE_DIR}/tenncor)
set(ROCNNET_DIR ${PROJECT_SOURCE_DIR}/rocnnet)

set(IO_DIR ${PROJECT_SOURCE_DIR}/tenncor_io)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

#>>>>>>>>>>>>>>>>>>>> DOXYGEN OPTION && SETUP <<<<<<<<<<<<<<<<<<<<
OPTION (BUILD_DOC "Build doxygen documents?" OFF)
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)
# adding doxygen
if(BUILD_DOC)
	add_subdirectory(${EXT_PROJECTS_DIR}/doxygen)
	if(DOXYGEN_FOUND)
		# set input and output files
		set(DOXYGEN_IN Doxyfile.in)
		set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

		message("Doxygen build started")
		configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
		add_customer_target(doc_doxygen ALL
			COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating API Documentation with Doxygen"
			VERBATIM)
	else (DOXYGEN_FOUND)
		message(WARNING "Doxygen need to be installed to generate the doxygen documetation")
	endif(DOXYGEN_FOUND)
endif(BUILD_DOC)

#>>>>>>>>>>>>>>>>>>>> BUILD TENNCOR LIBRARY <<<<<<<<<<<<<<<<<<<<
set(TENNCOR_INCLUDE ${TENNCOR_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/tenncor)
add_subdirectory(${TENNCOR_DIR})

#>>>>>>>>>>>>>>>>>>>> BUILD ROCNNET LIBRARY <<<<<<<<<<<<<<<<<<<<
set(ROCNNET_INCLUDE ${ROCNNET_DIR}/include)
FILE(GLOB_RECURSE ROCNNET_SRC ${ROCNNET_DIR}/src/*.cpp)

include_directories(${ROCNNET_INCLUDE} ${TENNCOR_INCLUDE})
add_library(rocnnet STATIC ${ROCNNET_SRC})
target_link_libraries(rocnnet tenncor)

#>>>>>>>>>>>>>>>>>>>> BUILD DEMO EXECUTABLE <<<<<<<<<<<<<<<<<<<<
if (LCOVERAGE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
endif(LCOVERAGE)
include_directories(${ROCNNET_INCLUDE})
add_executable(gd_demo ${ROCNNET_DIR}/demo/gd_demo.cpp)
target_link_libraries(gd_demo rocnnet)

#>>>>>>>>>>>>>>>>>>>> BUILD TENNCOR_IO LIBRARY FOR VISER <<<<<<<<<<<<<<<<<<<<
set(IO_INCLUDE ${IO_DIR}/include)
FILE(GLOB_RECURSE IO_SRC ${IO_DIR}/src/*.cpp)

# todo: uncomment to build
#include_directories(${IO_INCLUDE})
#add_library(tenncor_io SHARED ${IO_SRC})
#target_link_libraries(tenncor_io ${TENNCOR_LIB})

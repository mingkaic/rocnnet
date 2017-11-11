cmake_minimum_required(VERSION 2.8)
project(rocnnet C CXX)

find_package(Threads REQUIRED)

# turn off if swig is not set up correctly (setup is not trivial for all machines)
option(SWIG_DQN "Train and test DQN on python" ON)

set(RBUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BINARY_DIR ${RBUILD_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(TENNCOR_DIR ${PROJECT_SOURCE_DIR}/tenncor)
set(ROCNNET_DIR ${PROJECT_SOURCE_DIR}/rocnnet)
set(BOOSTPY_DIR ${PROJECT_SOURCE_DIR}/boostpy)

set(IO_DIR ${PROJECT_SOURCE_DIR}/tenncor_io)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# build dynamic path without setting off warnings
set(CMAKE_MACOSX_RPATH 1)

#>>>>>>>>>>>>>>>>>>>> DOXYGEN OPTION && SETUP <<<<<<<<<<<<<<<<<<<<
option(BUILD_DOC "Build doxygen documents?" OFF)
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

#>>>>>>>>>>>>>>>>>>>> VISUALIZATION OPTIONS <<<<<<<<<<<<<<<<<<<<
if (${CSV_RCD} OR ${RPC_RCD})
	# visualize tenncor edges
	message("ALLOWING VISUALIZATION OF GRAPH: WARNING (additional overhead)")
	add_definitions(-DEDGE_RCD)
endif()
if (${CSV_RCD})
	add_definitions(-DCSV_RCD)
elseif (${RPC_RCD})
	add_definitions(-DRPC_RCD)
endif()

#>>>>>>>>>>>>>>>>>>>> BUILD TENNCOR LIBRARY <<<<<<<<<<<<<<<<<<<<
set(TENNCOR_LIB tenncor)
set(TENNCOR_INCLUDE ${TENNCOR_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/tenncor)
add_subdirectory(${TENNCOR_DIR})

#>>>>>>>>>>>>>>>>>>>> BUILD ROCNNET LIBRARY <<<<<<<<<<<<<<<<<<<<
set(ROCNNET_INCLUDE ${ROCNNET_DIR}/include)
file(GLOB_RECURSE ROCNNET_SRC ${ROCNNET_DIR}/src/*.cpp)

set(ROCNNET_SLIB rocnnet_static)
include_directories(${ROCNNET_INCLUDE} ${TENNCOR_INCLUDE})
add_library(${ROCNNET_SLIB} STATIC ${ROCNNET_SRC})
target_link_libraries(${ROCNNET_SLIB} ${TENNCOR_LIB})

set(ROCNNET_DLIB rocnnet_shared)
include_directories(${ROCNNET_INCLUDE} ${TENNCOR_INCLUDE})
add_library(${ROCNNET_DLIB} SHARED ${ROCNNET_SRC})
target_link_libraries(${ROCNNET_DLIB} ${TENNCOR_LIB})

#>>>>>>>>>>>>>>>>>>>> BUILD BOOST-PYTHON INTERFACE <<<<<<<<<<<<<<<<<<<<
set(RBP_LIB rboostpy_lib)
set(RBP_INCLUDE ${BOOSTPY_DIR}/include)
add_subdirectory(${BOOSTPY_DIR})

find_package(PythonLibs REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS python numpy)

#>>>>>>>>>>>>>>>>>>>> BUILD DEMOS <<<<<<<<<<<<<<<<<<<<
set(DEMODIR ${PROJECT_SOURCE_DIR}/demo)
# DEMO 1: gradient descent network
if (LCOVERAGE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
endif(LCOVERAGE)
include_directories(${ROCNNET_INCLUDE})
add_executable(gd_demo ${DEMODIR}/gd_demo.cpp)
target_link_libraries(gd_demo ${ROCNNET_SLIB})

# DEMO 2: c++ deep q network rudimentary test
if (LCOVERAGE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
endif(LCOVERAGE)
include_directories(${ROCNNET_INCLUDE})
add_executable(dqn_demo ${DEMODIR}/dqn_demo.cpp)
target_link_libraries(dqn_demo ${ROCNNET_SLIB})

# DEMO 3: rbm
if (LCOVERAGE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
endif(LCOVERAGE)
include_directories(${ROCNNET_INCLUDE} ${RBP_INCLUDE} ${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
add_executable(rbm_demo ${DEMODIR}/rbm_demo.cpp)
target_link_libraries(rbm_demo ${ROCNNET_SLIB} ${RBP_LIB} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} )

# DEMO 4: dqn
if (LCOVERAGE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
endif(LCOVERAGE)
include_directories(${ROCNNET_INCLUDE} ${RBP_INCLUDE} ${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
add_executable(dbn_demo ${DEMODIR}/dbn_demo.cpp)
target_link_libraries(dbn_demo ${ROCNNET_SLIB} ${RBP_LIB} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} )

if (SWIG_DQN)
	# DEMO 2.1: pythonize deep q network with swig
	add_subdirectory(${ROCNNET_DIR}/ipython)
endif()

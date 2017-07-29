cmake_minimum_required(VERSION 2.8)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()

find_package(PythonLibs 2 REQUIRED)
message("Python version is ${PYTHONLIBS_VERSION_STRING}")
message("Python include found at ${PYTHON_INCLUDE_PATH}")
message("Python library found at ${PYTHON_LIBRARIES}")
message("Python executable found at ${PYTHON_EXECUTABLE}")
message("Python path is ${PYTHONPATH}")
include_directories(${PYTHON_INCLUDE_PATH} ${ROCNNET_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_SWIG_FLAGS "-keyword")
set(IFILE dqn_module.i)
# output must be same as library output to ensure .so and .py wrapping interface are together
set(CMAKE_SWIG_OUTDIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set_source_files_properties(${IFILE} PROPERTIES CPLUSPLUS ON)
swig_add_module(dqn python ${IFILE} dqn_agent.cpp)
swig_link_libraries(dqn ${PYTHON_LIBRARIES} ${ROCNNET_DLIB})

# manual relink python library on OSX
if(APPLE)
	add_custom_target(py_relink ALL
		DEPENDS _dqn
		COMMAND bash ${RBUILD_DIR}/scripts/mac_fix.sh ${PYTHON_LIBRARY} ${RBUILD_DIR}
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif(APPLE)
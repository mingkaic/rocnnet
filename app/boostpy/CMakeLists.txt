cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME rocboostpy)
project(rboostpy)

if (NOT RBUILD_DIR)
	set(RBUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if (NOT RBP_LIB)
	set(RBP_LIB RBP_LIB)
endif()

if (NOT RBP_INCLUDE)
	set(RBP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

set(CMAKE_BINARY_DIR ${RBUILD_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(PythonLibs REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS python numpy)

FILE(GLOB_RECURSE RBP_SRC src/*.cpp)

include_directories(${RBP_INCLUDE} ${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

add_library(${RBP_LIB} STATIC ${RBP_SRC})
target_link_libraries(${RBP_LIB} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(rboostpy_test tests/test_embedding.cpp ${RBP_SRC})
target_link_libraries(rboostpy_test ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})



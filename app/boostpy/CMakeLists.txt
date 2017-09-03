cmake_minimum_required(VERSION 3.9)
set(PROJECT_NAME rocboostpy)
project(rboostpy)

if (NOT RBUILD_DIR)
    set(RBUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(CMAKE_BINARY_DIR ${RBUILD_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(PythonLibs REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS python)

set(RBP_INCLUDE include)
FILE(GLOB_RECURSE RBP_SRC src/*.cpp)

include_directories(${RBP_INCLUDE} ${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

add_executable(rboostpy_test main.cpp ${RBP_SRC})
target_link_libraries(rboostpy_test ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

cmake_minimum_required(VERSION 3.6)
project(graph_gui C CXX)

set(BUILD_EXAMPLES OFF)
add_subdirectory(../ext/nodeeditor ../ext/nodeeditor)

set(CMAKE_AUTOMOC ON)
find_package(Qt5 COMPONENTS Core Widgets Gui OpenGL)

file(GLOB_RECURSE GUI_FILES src/*.cpp)
file(GLOB_RECURSE GUI_HDRS include/*.hpp)
include_directories(include ${TENNCOR_INCLUDE})
add_executable(graph_gui ${GUI_HDRS} ${GUI_FILES} ../tenncor/src/utils/utils.cpp)

target_include_directories(graph_gui INTERFACE "include")
target_link_libraries(graph_gui
	proto_grpc nodes)

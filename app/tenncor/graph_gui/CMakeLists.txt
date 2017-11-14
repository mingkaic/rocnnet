cmake_minimum_required(VERSION 2.8)
project(graph_gui C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

file(GLOB_RECURSE GUI_FILES ./*.cpp)
include_directories(${TENNCOR_INCLUDE})
add_executable(graph_gui ${GUI_FILES} ../src/utils/utils.cpp)
target_link_libraries(graph_gui proto_grpc)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Protobuf REQUIRED)
find_package(GRPC REQUIRED)

file(GLOB_RECURSE GRPC_FILES grpc/*.proto)
file(GLOB_RECURSE PROTO_FILES **/*.proto)

set(PROTO_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${PROTO_OUT_DIR})
include_directories(${PROTOBUF_INCLUDE_DIR} ${PROTO_OUT_DIR})

protobuf_generate_cpp(PROTO_SRC PROTO_HEADER ${PROTO_OUT_DIR} ${PROTO_FILES})
grpc_generate_cpp(GRPC_SRC GRPC_HEADER ${PROTO_OUT_DIR} ${GRPC_FILES})

add_library(proto ${PROTO_HEADER} ${PROTO_SRC})
add_library(grpc ${GRPC_HEADER} ${GRPC_SRC})

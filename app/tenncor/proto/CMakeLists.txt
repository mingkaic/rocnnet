include(FindProtobuf)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(GRPC REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIR})

protobuf_generate_cpp(PROTO_SRC PROTO_HEADER tenncor.proto)
file(GLOB_RECURSE GRPC_FILES grpc/*.proto)
grpc_generate_cpp(GRPC_SRC GRPC_HEADER ${GRPC_FILES})

add_library(proto ${PROTO_HEADER} ${PROTO_SRC})
add_library(grpc ${GRPC_HEADER} ${GRPC_SRC})

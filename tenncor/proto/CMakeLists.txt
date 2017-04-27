include(FindProtobuf)

include_directories(${PROTOBUF_INCLUDE_DIR})
protobuf_generate_cpp(PROTO_SRC PROTO_HEADER tenncor.proto)
add_library(proto ${PROTO_HEADER} ${PROTO_SRC})

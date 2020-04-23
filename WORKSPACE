workspace(name = "com_github_mingkaic_rocnnet")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "com_github_mingkaic_tenncor",
    remote = "https://github.com/mingkaic/tenncor",
    commit = "8cd9536fef5b28dc3212bda9769995444878913d",
)

# ===== load all tenncor build dependencies ===

load("@com_github_mingkaic_tenncor//third_party:all.bzl", tenncor_deps="dependencies")
tenncor_deps()

load("@com_github_mingkaic_cppkg//:cppkg.bzl", cppkg_deps="dependencies")
cppkg_deps()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "com_google_protobuf_custom",
    sha256 = "a19dcfe9d156ae45d209b15e0faed5c7b5f109b6117bfc1974b6a7b98a850320",
    strip_prefix = "protobuf-3.7.0",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.7.0.tar.gz"],
)

workspace(name = "com_github_mingkaic_rocnnet")

# === load tenncor dependencies ===

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
git_repository(
    name = "com_github_mingkaic_tenncor",
    remote = "https://github.com/mingkaic/tenncor",
    commit = "b1a308017b7b3e3f04d9613665538fb600546d96",
)

load("@com_github_mingkaic_tenncor//third_party:all.bzl", tenncor_deps="dependencies")
tenncor_deps()

# === load cppkg dependencies ===

load("@com_github_mingkaic_cppkg//third_party:all.bzl", cppkg_deps="dependencies")
cppkg_deps(excludes=["gtest"])

# === boost dependencies ===

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

# === load grpc depedencies ===

# common dependencies
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

# python dependencies
load("@rules_proto_grpc//python:repositories.bzl", rules_proto_grpc_python_repos="python_repos")
rules_proto_grpc_python_repos()

load("@rules_python//python:repositories.bzl", "py_repositories")
load("@rules_python//python:pip.bzl", "pip_repositories", "pip_import")
py_repositories()
pip_repositories()
pip_import(
    name = "rules_proto_grpc_py2_deps",
    python_interpreter = "python", # Replace this with the platform specific Python 2 name, or remove if not using Python 2
    requirements = "@rules_proto_grpc//python:requirements.txt",
)
pip_import(
    name = "rules_proto_grpc_py3_deps",
    python_interpreter = "python3",
    requirements = "@rules_proto_grpc//python:requirements.txt",
)
load("@rules_proto_grpc_py2_deps//:requirements.bzl", pip2_install="pip_install")
load("@rules_proto_grpc_py3_deps//:requirements.bzl", pip3_install="pip_install")
pip2_install()
pip3_install()

# === load pybind dependencies ===

load("@com_github_pybind_bazel//:python_configure.bzl", "python_configure")
python_configure(name="local_config_python")

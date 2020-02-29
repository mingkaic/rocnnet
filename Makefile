
CC := clang


print_vars:
	@echo "CC: " $(CC)

rocnnet_py_build:
	bazel build --config $(CC)_eigen_optimal @com_github_mingkaic_tenncor//:tenncor_py

rocnnet_py_export: rocnnet_py_build
	mkdir -p bin
	cp -f bazel-bin/external/com_github_mingkaic_tenncor/*.so ./bin

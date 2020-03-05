
CC := clang

IMAGE_REPO := mkaichen
IMAGE_TAG := latest

print_vars:
	@echo "CC: " $(CC)

.PHONY: rocnnet_py_build
rocnnet_py_build:
	bazel build --config $(CC)_eigen_optimal @com_github_mingkaic_tenncor//:tenncor_py

.PHONY: rocnnet_py_export
rocnnet_py_export: rocnnet_py_build
	cp -f bazel-bin/external/com_github_mingkaic_tenncor/*.so .
	cp bazel-rocnnet/external/com_github_mingkaic_tenncor/cfg/optimizations.json cfg/.

.PHONY: build_test_image
build_test_image:
	docker build -f Dockerfile.test -t ${IMAGE_REPO}/rocnnet-test:${IMAGE_TAG} .

.PHONY: push_test_image
push_test_image:
	docker push ${IMAGE_REPO}/rocnnet-test:${IMAGE_TAG}

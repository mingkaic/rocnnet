
CC := clang

IMAGE_REPO := mkaichen
IMAGE_TAG := latest

print_vars:
	@echo "CC: " $(CC)

.PHONY: external_export
external_export: tenncor_py_export onnxds_py_export

.PHONY: tenncor_py_build
tenncor_py_build:
	bazel build --config $(CC)_eigen_optimal @com_github_mingkaic_tenncor//tenncor:tenncor_py
	bazel build --config $(CC)_eigen_optimal @com_github_mingkaic_tenncor//extenncor:extenncor

.PHONY: tenncor_py_export
tenncor_py_export: tenncor_py_build
	mkdir -p onnxds
	cp -f bazel-bin/external/com_github_mingkaic_tenncor/tenncor/*.so .
	cp -r bazel-rocnnet/external/com_github_mingkaic_tenncor/extenncor .
	cp bazel-rocnnet/external/com_github_mingkaic_tenncor/cfg/optimizations.json cfg

.PHONY: onnxds_py_build
onnxds_py_build:
	bazel build @com_github_mingkaic_onnxds//onnxds:read_dataset

.PHONY: onnxds_py_export
onnxds_py_export: onnxds_py_build
	mkdir -p onnxds
	cp -r bazel-rocnnet/external/com_github_mingkaic_onnxds/onnxds .
	cp -r bazel-rocnnet/external/com_github_mingkaic_onnxds/onnx .

.PHONY: build_test_image
build_test_image:
	docker build -f Dockerfile.test -t ${IMAGE_REPO}/rocnnet-test:${IMAGE_TAG} .

.PHONY: push_test_image
push_test_image:
	docker push ${IMAGE_REPO}/rocnnet-test:${IMAGE_TAG}

.PHONY: generate_cifar10
generate_cifar10:
	bazel run @com_github_mingkaic_onnxds//:tfgen_dataset -- --out /tmp/cifar_train.onnx --split TRAIN --batch 5 cifar10
	bazel run @com_github_mingkaic_onnxds//:tfgen_dataset -- --out /tmp/cifar_test.onnx --split TEST --batch 1 cifar10
	mv /tmp/cifar_train.onnx models
	mv /tmp/cifar_test.onnx models

.PHONY: clean
clean:
	rm *.so
	rm -rf extenncor

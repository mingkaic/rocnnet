//
// Created by Mingkai Chen on 2016-08-29.
//

#include "tensor/tensor.hpp"

TEST(TENSOR, tensor_construct) {
	nnet::tensor<double> t1; // no shape -> equiv to scalar
	EXPECT_FALSE(t1.is_alloc());
	EXPECT_EQ(0, t1.n_dims());
	EXPECT_EQ(0, t1.n_elems());

	nnet::tensor<double> t2(
		(nnet::tensorshape())); // undefined shape
	EXPECT_FALSE(t2.is_alloc());
	EXPECT_EQ(0, t2.n_dims());
	EXPECT_EQ(0, t2.n_elems());

	nnet::tensor<double> t3(
		(std::vector<size_t>){3, 0}); // defined with unknown
	EXPECT_FALSE(t3.is_alloc());
	EXPECT_EQ(2, t3.n_dims());
	EXPECT_EQ(0, t3.n_elems());

	// allocation constructions
	nnet::tensor<double> tgood(std::vector<size_t>{2, 3});
	EXPECT_TRUE(tgood.is_alloc());
	EXPECT_EQ(2, tgood.n_dims());
	EXPECT_EQ(6, tgood.n_elems());

	// attribute?
}


// TODO variable clone tester (test for deep copy)


TEST(TENSOR, tensor_allocate) {
	// unknown
	nnet::tensor<double> liberal((nnet::tensorshape()));
	// 3 by ? matrix
	nnet::tensor<double> partial((std::vector<size_t>){3, 0});
	// 3 by 3 matrix
	nnet::tensor<double> exact((std::vector<size_t>){3, 3});

	nnet::iallocator locker = new nnet::ram_alloc();
	// allocate from allowed shape
	EXPECT_DEATH({ liberal.allocate(locker); }, ".*");
	EXPECT_DEATH({ partial.allocate(locker); }, ".*");
	/* EXPECT_LIFE */ exact.allocate(locker);

	// allocate shape
	std::vector<nnet::tensorshape> shapes = {
		// fits liberal only
		(std::vector<size_t>){1, 3},
		// fits liberal and partial
		(std::vector<size_t>){3, 2},
		// fits any
		(std::vector<size_t>){3, 3}
	};

	for (nnet::tensorshape s : shapes) {
		liberal.allocate(locker, s);
	}

	EXPECT_DEATH({ partial.allocate(locker, shapes[0]); }, ".*");
	for (size_t i = 1; i < shapes.size(); i++) {
		partial.allocate(locker, shapes[i]);
	}

	for (size_t i = 0; i < shapes.size()-1; i++) {
		EXPECT_DEATH({ exact.allocate(locker, shapes[i]); }, ".*");
	}
	exact.allocate(locker, shapes[shapes.size()-1]);
}
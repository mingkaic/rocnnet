//
//  test_variable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../shared/utils.hpp"

TEST(VARIABLE, tensor_construct) {
    nnet::tensor<double> t1; // no shape -> equiv to scalar
    EXPECT_FALSE(t1.is_alloc());
    EXPECT_EQ(0, t1.n_dims());
    EXPECT_EQ(0, t1.n_elems());

    nnet::tensor<double> t2(
        (nnet::tensor_shape())); // undefined shape
    EXPECT_FALSE(t2.is_alloc());
    EXPECT_EQ(0, t2.n_dims());
    EXPECT_EQ(0, t2.n_elems());

    nnet::tensor<double> t3(
        (std::vector<size_t>){3, 0}); // defined with unknown
    EXPECT_FALSE(t3.is_alloc());
    EXPECT_EQ(2, t3.n_dims());
    EXPECT_EQ(0, t3.n_elems());

    nnet::memory_alloc alloc;
    // allocation constructions
    nnet::tensor<double> tgood(alloc,
        (std::vector<size_t>){2, 3});
    EXPECT_TRUE(tgood.is_alloc());
    EXPECT_EQ(2, tgood.n_dims());
    EXPECT_EQ(6, tgood.n_elems());

    // must be fully defined
    EXPECT_DEATH({
        nnet::tensor<double> tbad1(alloc,
            (std::vector<size_t>){0, 1});
    }, ".*");

    EXPECT_DEATH({
        nnet::tensor<double> tbad2(alloc,
            nnet::tensor_shape());
    }, ".*");

    // attribute?
}


// TODO variable clone tester (test for deep copy)


TEST(VARIABLE, tensor_allocate) {
    // unknown
    nnet::tensor<double> liberal((nnet::tensor_shape()));
    // 3 by ? matrix
    nnet::tensor<double> partial((std::vector<size_t>){3, 0});
    // 3 by 3 matrix
    nnet::tensor<double> exact((std::vector<size_t>){3, 3});

    nnet::memory_alloc locker;
    // allocate from allowed shape
    EXPECT_DEATH({ liberal.allocate(locker); }, ".*");
    EXPECT_DEATH({ partial.allocate(locker); }, ".*");
    /* EXPECT_LIFE */ exact.allocate(locker);

    // allocate shape
    std::vector<nnet::tensor_shape> shapes = {
        // fits liberal only
        (std::vector<size_t>){1, 3},
        // fits liberal and partial
        (std::vector<size_t>){3, 2},
        // fits any
        (std::vector<size_t>){3, 3}
    };

    for (nnet::tensor_shape s : shapes) {
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


TEST(VARIABLE, const_init) {
    const double constant = (double) rand();
    nnet::const_init<double> init(constant);

    std::shared_ptr<nnet::variable<double> > cvar =
    	std::make_shared<nnet::variable<double> >(std::vector<size_t>{3, 3}, init, "constant_arr");

    cvar->initialize();
    EXPOSE_PTR ex = EXPOSE(cvar);
    std::vector<double> raw = ex->get_raw();

    ASSERT_EQ(raw.size(), 9);

    for (double elem : raw) {
        EXPECT_EQ(elem, constant);
    }
}


TEST(VARIABLE, get_index) {
    const double constant = (double) rand();
    nnet::const_init<double> init(constant);
    size_t nrows = 3;
    size_t ncols = 4;
    size_t ndeps = 5; // depth

	std::shared_ptr<nnet::variable<double> > cvar =
			std::make_shared<nnet::variable<double> >(std::vector<size_t>{nrows, ncols, ndeps}, init, "constant_arr");

    cvar->initialize();

	EXPOSE_PTR ex = EXPOSE(cvar);
    nnet::tensor<double> raw = ex->eval();

    ASSERT_EQ(raw.n_elems(), nrows*ncols*ndeps);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            for (size_t k = 0; k < ndeps; k++) {
                EXPECT_EQ(constant, raw.get({i,j,k}));
            }
        }
    }
}


TEST(VARIABLE, random_init) {
    nnet::random_uniform<double> init(0.0, 1.0);

	std::shared_ptr<nnet::variable<double> > rvar =
			std::make_shared<nnet::variable<double> >(std::vector<size_t>{5, 5}, init, "random_arr");

    rvar->initialize();

	EXPOSE_PTR ex = EXPOSE(rvar);
    std::vector<double> raw = ex->get_raw();

    ASSERT_EQ(raw.size(), 25);

	double sum = 0;
    for (double elem : raw) {
		sum += elem;
        EXPECT_GT(elem, 0);
        EXPECT_LT(elem, 1);
    }
    ASSERT_GT(sum/raw.size(), 0);
}


TEST(VARIABLE, placeholder) {
    const size_t insize = 20;
    nnet::PLACEHOLDER_PTR<double> invar = MAKE_PLACEHOLDER((std::vector<size_t>{1, insize}), "in");
    std::vector<double> sample;
    for (size_t i = 0; i < insize; i++) {
        sample.push_back(rand());
    }
    *invar = sample;

    EXPOSE_PTR ex = EXPOSE(invar);
    std::vector<double> raw = ex->get_raw();

    ASSERT_EQ(raw.size(), insize);

    for (size_t i = 0; i < insize; i++) {
        EXPECT_EQ(sample[i], raw[i]);
    }
}

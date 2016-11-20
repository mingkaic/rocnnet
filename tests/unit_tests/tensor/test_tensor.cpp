//
// Created by Mingkai Chen on 2016-08-29.
//

#include <algorithm>
#include "tensor_test_util.h"
#include "tensor/tensor.hpp"
#include "tensor/tensor_op.hpp"
#include "graph/initializer.hpp"

static nnet::random_uniform<double> rinit(-3.4, 3.14);
    
// Behavior B100
TEST(TENSOR, AllocateOnConstruct_B100)
{
    nnet::tensor<double, nnet::ram_alloc> pure_good(std::vector<size_t>{1, 2, 3});
    nnet::tensor<double, nnet::ram_alloc> scalar(4);
    EXPECT_TRUE(pure_good.is_alloc());
    EXPECT_TRUE(scalar.is_alloc());
	EXPECT_EQ(3, pure_good.n_dims());
	EXPECT_EQ(6, pure_good.n_elems());
	EXPECT_EQ(1, scalar.n_dims());
	EXPECT_EQ(1, scalar.n_elems());
}


// Inverse Test of B100
TEST(TENSOR, Construct)
{
	nnet::tensor<double> incom;
	nnet::tensor<double> pcom(std::vector<size_t>{3, 0});
	EXPECT_FALSE(incom.is_alloc());
	EXPECT_EQ(0, incom.n_dims());
	EXPECT_EQ(0, incom.n_elems());
	EXPECT_FALSE(pcom.is_alloc());
	EXPECT_EQ(2, pcom.n_dims());
	EXPECT_EQ(0, pcom.n_elems());
}


// COPY
TEST(TENSOR, Copy)
{
	nnet::tensor<double> incom;
	nnet::tensor<double> pcom(std::vector<size_t>{3, 0});
    nnet::tensor<double> com(std::vector<size_t>{1, 2, 3});
    
    rinit(incom);
    rinit(pcom);
    rinit(com);
    
    nnet::tensor<double>* fresh_incom = incom.clone();
    nnet::tensor<double>* fresh_pcom = pcom.clone();
    nnet::tensor<double>* fresh_com = com.clone();
    
	nnet::tensor<double> a;
	nnet::tensor<double> b;
	nnet::tensor<double> c;
	a = incom;
	b = pcom;
	c = com;
	
	// equate a, b, c, and fresh tensors
	// shape equate
	nnet::tensorshape fresh_ints = fresh_incom->get_shape();
	nnet::tensorshape fresh_pts = fresh_pcom->get_shape();
	nnet::tensorshape fresh_ts = fresh_com->get_shape();
	nnet::tensorshape ats = a.get_shape();
	nnet::tensorshape bts = b.get_shape();
	nnet::tensorshape cts = c.get_shape();
	
	ASSERT_TRUE(tensorshape_equal(fresh_ints, incom.get_shape()));
	ASSERT_TRUE(tensorshape_equal(fresh_pts, pcom.get_shape()));
	ASSERT_TRUE(tensorshape_equal(fresh_ts, com.get_shape()));
	ASSERT_TRUE(tensorshape_equal(ats, a.get_shape()));
	ASSERT_TRUE(tensorshape_equal(bts, b.get_shape()));
	ASSERT_TRUE(tensorshape_equal(cts, c.get_shape()));
	
	ASSERT_DEATH(nnet::expose<double>(&incom), ".*");
	ASSERT_DEATH(nnet::expose<double>(fresh_incom), ".*");
	ASSERT_DEATH(nnet::expose<double>(&a), ".*");
	ASSERT_DEATH(nnet::expose<double>(&pcom), ".*");
	ASSERT_DEATH(nnet::expose<double>(fresh_pcom), ".*");
	ASSERT_DEATH(nnet::expose<double>(&b), ".*");
	
	std::vector<double> expect = nnet::expose<double>(&com);
	std::vector<double> raw1 = nnet::expose<double>(fresh_com);
	std::vector<double> raw2 = nnet::expose<double>(&c);
	
	std::equal(expect.begin(), expect.end(), raw1.begin());
	std::equal(expect.begin(), expect.end(), raw2.begin());

    delete fresh_incom;
    delete fresh_pcom;
    delete fresh_com;
}


// Behavior B101
TEST(TENSOR, Allocate_B101)
{
	nnet::tensor<double> t1(std::vector<size_t>{1, 2, 3});
	nnet::tensor<double> u1(std::vector<size_t>{0, 1, 2});
	// exactly compatible except, input is undefined
	EXPECT_DEATH(t1.allocate(std::vector<size_t>{0, 2, 3}), ".*");
	// Either equivalent shape
	t1.allocate(std::vector<size_t>{1, 2, 3});
	// Or none at all
	t1.allocate();
	// Undefined tensors absolutely require a fully defined shape on allocation
	EXPECT_DEATH(u1.allocate(std::vector<size_t>{0, 2, 3}), ".*");
	EXPECT_DEATH(u1.allocate(), ".*");
}

// Behavior B102
TEST(TENSOR, AllocateCompatible_B102)
{
	nnet::tensor<double> t1(std::vector<size_t>{1, 2, 3});
	// same number of elements, but different shape
	EXPECT_DEATH(t1.allocate(std::vector<size_t>{3, 2, 1}), ".*");
	// technically the same except different rank
	EXPECT_DEATH(t1.allocate(std::vector<size_t>{1, 2, 3, 1}), ".*");
}

// Behavior B103
TEST(TENSOR, Operation_B103)
{
	const size_t elems = 6;
	const nnet::tensorshape unified = std::vector<size_t>{2, 3};
	nnet::tensor<double> A(unified);
	nnet::tensor<double> B(unified);
	nnet::tensor_op<double> op([&elems](double*& dest,std::vector<const double*> srcs)
	{
		std::memset(dest, 0, elems * sizeof(double));
		for (const double* src: srcs)
		{
			for (size_t i = 0; i < elems; i++)
			{
				dest[i] += src[i];
			}
		}
	});
	op.set_shape(unified);
	op(std::vector<nnet::tensor<double>*>{&A, &B});
	rinit(A);
	rinit(B);
	// expose shape and equivalent to 
	std::vector<double> Ares = nnet::expose<double>(&A);
	std::vector<double> Bres = nnet::expose<double>(&B);
	std::vector<double> expectOut;
	
	for (size_t i = 0; i < Ares.size(); i++)
	{
		expectOut.push_back(Ares[i] + Bres[i]);
	}
	std::vector<double> res = nnet::expose<double>(&op);
	std::equal(expectOut.begin(), expectOut.end(), res.begin());
}


// ACCESSORS
TEST(TENSOR, GetIndex) {
	size_t x = 2, y = 3, z = 4;
	
	const double constant = (double) rand();
	nnet::const_init<double> init(constant);
	nnet::tensor<double> ten(std::vector<size_t>{x, y, z});
	init(ten);

	for (size_t i = 0; i < x; i++) {
		for (size_t j = 0; j < y; j++) {
			for (size_t k = 0; k < z; k++) {
				EXPECT_EQ(constant, ten.get({i,j,k}));
			}
		}
	}
}
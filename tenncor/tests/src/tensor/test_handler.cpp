//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"
#include "util_test.h"

#include "tensor/tensor_handler.hpp"
#include "mocks/mock_tensor.h"
// avoid mock tensor to prevent its innate random initialization


using namespace nnet;


//#define DISABLE_HANDLER_TEST
#ifndef DISABLE_HANDLER_TEST


static double SUPERMARK = 1;


static void forward (double* out, const tensorshape& outs,
	std::vector<const double*>& ins,
	std::vector<tensorshape>& inshapes)
{
	assert(inshapes.size() == ins.size());
	assert(inshapes.size() == 2);
	size_t n = inshapes[0].n_elems();
	size_t m = inshapes[1].n_elems();
	std::memset(out, 0, sizeof(double) * outs.n_elems());
	for (size_t i = 0; i < n; i++)
	{
		out[i] = ins[0][i];
	}
	for (size_t i = 0; i < m; i++)
	{
		out[i+n] = ins[1][i];
	}
}


static tensorshape shaper (std::vector<tensorshape> args)
{
	return args[0].concatenate(args[1]);
}


static void marked_forward (double* out, const tensorshape& outs,
	std::vector<const double*>&, std::vector<tensorshape>&)
{
	std::memset(out, 0, sizeof(double) * outs.n_elems());
	out[0] = SUPERMARK;
}


static tensorshape marked_shape (std::vector<tensorshape>)
{
	return std::vector<size_t>{(size_t)SUPERMARK};
}


// cover transfer_function
// operator ()
TEST(HANDLER, Transfer_C000)
{
	tensorshape c1 = random_shape();
	tensorshape c2 = random_shape();
	tensorshape resshape = c1.concatenate(c2);
	mock_tensor arg1(c1);
	mock_tensor arg2(c2);
	tensor<double> good(resshape);
	tensor<double> bad(c1);
	std::vector<const tensor<double>*> args;
	args.push_back(&arg1);
	args.push_back(&arg2);

	transfer_func<double> tf(shaper, forward);
	EXPECT_THROW(tf(bad, args), std::exception);
	tf(good, args);
	std::vector<double> d1 = arg1.expose();
	std::vector<double> d2 = arg2.expose();
	std::vector<double> res = good.expose();
	size_t n = c1.n_elems();
	size_t m = c2.n_elems();
	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(res[i], d1[i]);
	}
	for (size_t i = 0; i < m; i++)
	{
		EXPECT_EQ(res[n+i], d2[i]);
	}
	for (size_t i = n+m, n = resshape.n_elems(); i < n; i++)
	{
		EXPECT_EQ(0, res[i]);
	}
}


// cover const_init
// operator ()
TEST(HANDLER, Constant_C001)
{
	double scalar = FUZZ::getDouble(1)[0];
	const_init<double> ci(scalar);
	tensorshape shape = random_shape();
	tensor<double> block(shape);
	ci(block);

	std::vector<double> v = block.expose();

	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_EQ(scalar, v[i]);
	}
}


// cover rand_uniform
// operator ()
TEST(HANDLER, Random_C002)
{
	double lo = FUZZ::getDouble(1, {127182, 12921231412323})[0];
	double hi = lo+1;
	double high = FUZZ::getDouble(1, {lo*2, lo*3+50})[0];
	rand_uniform<double> ri1(lo, hi);
	rand_uniform<double> ri2(lo, high);
	tensorshape shape = random_shape();
	tensor<double> block1(shape);
	tensor<double> block2(shape);
	ri1(block1);
	ri2(block2);

	std::vector<double> v1 = block1.expose();
	std::vector<double> v2 = block2.expose();

	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(lo, v1[i]);
		EXPECT_GE(hi, v1[i]);
	}
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(lo, v2[i]);
		EXPECT_GE(high, v2[i]);
	}
}


// cover transfer_func, const_init, rand_uniform
// copy constructor and assignment
TEST(HANDLER, Copy_C003)
{
	SUPERMARK = 0;
	transfer_func<double> tfassign(marked_shape, marked_forward);
	const_init<double> ciassign(0);
	rand_uniform<double> riassign(0, 1);

	SUPERMARK = FUZZ::getDouble(1, {15, 117})[0];
	double scalar = FUZZ::getDouble(1)[0];
	double low = FUZZ::getDouble(1, {23, 127})[0];
	double high = FUZZ::getDouble(1, {low*2, low*3+50})[0];
	transfer_func<double> tf(marked_shape, marked_forward);
	const_init<double> ci(scalar);
	rand_uniform<double> ri(low, high);

	transfer_func<double>* tfcpy = tf.clone();
	const_init<double>* cicpy = ci.clone();
	rand_uniform<double>* ricpy = ri.clone();

	tfassign = tf;
	ciassign = ci;
	riassign = ri;

	tensorshape shape = random_shape();
	tensor<double> tscalar(0);
	tensor<double> tblock(shape);
	tensor<double> ttransf(std::vector<size_t>{(size_t) SUPERMARK});
	(*tfcpy)(ttransf, {});
	std::vector<double> transfv = ttransf.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv.size());
	EXPECT_EQ(SUPERMARK, transfv[0]);

	(*cicpy)(tscalar);
	EXPECT_EQ(scalar, tscalar.expose()[0]);

	(*ricpy)(tblock);
	std::vector<double> v = tblock.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v[i]);
		EXPECT_GE(high, v[i]);
	}

	tensor<double> tscalar2(0);
	tensor<double> tblock2(shape);
	tensor<double> ttransf2(std::vector<size_t>{(size_t) SUPERMARK});
	tfassign(ttransf2, {});
	std::vector<double> transfv2 = ttransf2.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv2.size());
	EXPECT_EQ(SUPERMARK, transfv2[0]);

	ciassign(tscalar2);
	EXPECT_EQ(scalar, tscalar2.expose()[0]);

	riassign(tblock2);
	std::vector<double> v2 = tblock2.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v2[i]);
		EXPECT_GE(high, v2[i]);
	}

	itensor_handler<double>* interface_ptr = &tf;
	itensor_handler<double>* resptr = interface_ptr->clone();
	EXPECT_NE(nullptr, dynamic_cast<transfer_func<double>*>(resptr));
	delete resptr;

	interface_ptr = &ci;
	resptr = interface_ptr->clone();
	EXPECT_NE(nullptr, dynamic_cast<const_init<double>*>(resptr));
	delete resptr;

	interface_ptr = &ri;
	resptr = interface_ptr->clone();
	EXPECT_NE(nullptr, dynamic_cast<rand_uniform<double>*>(resptr));
	delete resptr;

	delete tfcpy;
	delete cicpy;
	delete ricpy;
}


// cover transfer_func, const_init, rand_uniform
// move constructor and assignment
TEST(HANDLER, Move_C003)
{
	SUPERMARK = 0;
	transfer_func<double> tfassign(marked_shape, marked_forward);
	const_init<double> ciassign(0);
	rand_uniform<double> riassign(0, 1);

	SUPERMARK = FUZZ::getDouble(1, {119, 221})[0];
	double scalar = FUZZ::getDouble(1)[0];
	double low = FUZZ::getDouble(1, {23, 127})[0];
	double high = FUZZ::getDouble(1, {low*2, low*3+50})[0];
	transfer_func<double> tf(marked_shape, marked_forward);
	const_init<double> ci(scalar);
	rand_uniform<double> ri(low, high);

	transfer_func<double> tfmv(std::move(tf));
	const_init<double> cimv(std::move(ci));
	rand_uniform<double> rimv(std::move(ri));

	tensorshape shape = random_shape();
	tensor<double> tscalar(0);
	tensor<double> tblock(shape);
	tensor<double> ttransf(std::vector<size_t>{(size_t) SUPERMARK});
	tfmv(ttransf, {});
	std::vector<double> transfv = ttransf.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv.size());
	EXPECT_EQ(SUPERMARK, transfv[0]);

	cimv(tscalar);
	EXPECT_EQ(scalar, tscalar.expose()[0]);

	rimv(tblock);
	std::vector<double> v = tblock.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v[i]);
		EXPECT_GE(high, v[i]);
	}

	tfassign = std::move(tfmv);
	ciassign = std::move(cimv);
	riassign = std::move(rimv);

	tensor<double> tscalar2(0);
	tensor<double> tblock2(shape);
	tensor<double> ttransf2(std::vector<size_t>{(size_t) SUPERMARK});
	tfassign(ttransf2, {});
	std::vector<double> transfv2 = ttransf2.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv2.size());
	EXPECT_EQ(SUPERMARK, transfv2[0]);

	ciassign(tscalar2);
	EXPECT_EQ(scalar, tscalar2.expose()[0]);

	riassign(tblock2);
	std::vector<double> v2 = tblock2.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v2[i]);
		EXPECT_GE(high, v2[i]);
	}

	// tf, ci, ri, tfmv, cimv, and rimv should be undefined
	tensor<double> tscalar3(0);
	tensor<double> tblock3(shape);
	tensor<double> ttransf3(std::vector<size_t>{(size_t) SUPERMARK});

	// moving function and resulting in undefined is a standard-dependent behavior
	// todo: decide on a more standard-independent way of testing moving function behavior
//	EXPECT_DEATH(tf(ttransf3, {}), ".*");
//	EXPECT_DEATH(ci(tscalar3), ".*");
//	EXPECT_DEATH(ri(tblock3), ".*");
//	EXPECT_DEATH(tfmv(ttransf3, {}), ".*");
//	EXPECT_DEATH(cimv(tscalar3), ".*");
//	EXPECT_DEATH(rimv(tblock3), ".*");
}


#endif /* DISABLE_HANDLER_TEST */

#endif /* DISABLE_TENSOR_MODULE_TESTS */

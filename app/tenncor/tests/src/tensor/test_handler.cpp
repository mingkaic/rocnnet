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


#ifndef DISABLE_HANDLER_TEST


// todo: remove this bad practice, maybe deterministically mark shapes and data if necessary
static double SUPERMARK = 1;



static void marked_forward (double* dest, std::vector<const double*>, nnet::shape_io shape)
{
	std::fill(dest, dest + shape.outs_.n_elems(), SUPERMARK);
}


static tensorshape shaper (std::vector<tensorshape> args)
{
	std::vector<size_t> s0 = args[0].as_list();
	std::vector<size_t> s1 = args[1].as_list();
	std::vector<size_t> compress;
	for (size_t i = 0, n = std::min(s0.size(), s1.size()); i < n; i++)
	{
		compress.push_back(std::min(s0[i], s1[i]));
	}
	return compress;
}


// cover transfer_function
// operator ()
TEST(HANDLER, Transfer_C000)
{
	FUZZ::reset_logger();
	tensorshape c1 = random_def_shape();
	tensorshape c2 = random_def_shape();
	tensorshape resshape = shaper({c1, c2});
	mock_tensor arg1(c1);
	mock_tensor arg2(c2);
	tensor<double> good(resshape);
	std::vector<const tensor<double>*> args = { &arg1, &arg2 };

	transfer_func<double> tf(adder);
	tf(good, args);
	std::vector<double> d1 = arg1.expose();
	std::vector<double> d2 = arg2.expose();
	std::vector<double> res = good.expose();
	size_t n = c1.n_elems();
	size_t m = c2.n_elems();
	size_t l = resshape.n_elems();
	for (size_t i = 0, k = std::min(std::min(n, m), l); i < k; i++)
	{
		EXPECT_EQ(res[i], d1[i] + d2[i]);
	}
}


// cover shape_extracter
// operator ()
TEST(HANDLER, ShapeExtractor_C001)
{
	FUZZ::reset_logger();
	tensorshape c1 = random_def_shape();
	tensorshape resshape = std::vector<size_t>{ c1.rank() };
	tensor<double> good(resshape);
	std::vector<tensorshape> args = { c1 };

	shape_extracter<double> se([](tensorshape& in) -> std::vector<size_t>
	{
		return in.as_list();
	});
	se(good, args);
	std::vector<size_t> sexpect = c1.as_list();
	std::vector<double> expect(sexpect.begin(), sexpect.end());
	std::vector<double> res = good.expose();
	EXPECT_TRUE(std::equal(expect.begin(), expect.end(), res.begin()));
}


// cover const_init
// operator ()
TEST(HANDLER, Constant_C002)
{
	FUZZ::reset_logger();
	double scalar = FUZZ::getDouble(1, "scalar")[0];
	const_init<double> ci(scalar);
	tensorshape shape = random_def_shape();
	tensor<double> block(shape);
	ci(block);

	std::vector<double> v = block.expose();

	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_EQ(scalar, v[i]);
	}
}


// cover rand_uniform, rand_normal
// operator ()
TEST(HANDLER, Random_C003)
{
	FUZZ::reset_logger();
	double lo = FUZZ::getDouble(1, "lo", {127182, 12921231412323})[0];
	double hi = lo+1;
	double high = FUZZ::getDouble(1, "high", {lo*2, lo*3+50})[0];
	double mean = FUZZ::getDouble(1, "mean", {-13, 23})[0];
	double variance = FUZZ::getDouble(1, "variance", {1, 32})[0];
	rand_uniform<double> ri1(lo, hi);
	rand_uniform<double> ri2(lo, high);
	rand_normal<double> rn(mean, variance);
	tensorshape shape = random_def_shape();
	tensor<double> block1(shape);
	tensor<double> block2(shape);
	tensor<double> block3(shape);
	ri1(block1);
	ri2(block2);
	rn(block3);

	std::vector<double> v1 = block1.expose();
	std::vector<double> v2 = block2.expose();
	std::vector<double> v3 = block3.expose();

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
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance / 2 of the normal
	double norm_mean = std::accumulate(v3.begin(), v3.end(), 0) / (double) v3.size();
	EXPECT_GT(norm_mean, mean - variance);
	EXPECT_LT(norm_mean, mean + variance);
}


// cover transfer_func, const_init, rand_uniform
// copy constructor and assignment
TEST(HANDLER, Copy_C004)
{
	FUZZ::reset_logger();
	SUPERMARK = 0;
	transfer_func<double> tfassign(marked_forward);
	const_init<double> ciassign(0);
	rand_uniform<double> riassign(0, 1);
	rand_normal<double> niassign;

	SUPERMARK = FUZZ::getDouble(1, "SUPERMARK", {15, 117})[0];
	double scalar = FUZZ::getDouble(1, "scalar")[0];
	double low = FUZZ::getDouble(1, "low", {23, 127})[0];
	double high = FUZZ::getDouble(1, "high", {low*2, low*3+50})[0];
	double mean = FUZZ::getDouble(1, "mean", {-13, 23})[0];
	double variance = FUZZ::getDouble(1, "variance", {1, 32})[0];
	transfer_func<double> tf(marked_forward);
	const_init<double> ci(scalar);
	rand_uniform<double> ri(low, high);
	rand_normal<double> ni(mean, variance);

	transfer_func<double>* tfcpy = tf.clone();
	const_init<double>* cicpy = ci.clone();
	rand_uniform<double>* ricpy = ri.clone();
	rand_normal<double>* nicpy = ni.clone();

	tfassign = tf;
	ciassign = ci;
	riassign = ri;
	niassign = ni;

	tensorshape shape = random_def_shape();
	tensor<double> tscalar(0);
	tensor<double> tblock(shape);
	tensor<double> tblock_norm(shape);
	tensor<double> ttransf(std::vector<size_t>{(size_t) SUPERMARK});
	std::vector<const tensor<double>*> empty_args;
	(*tfcpy)(ttransf, empty_args);
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

	(*nicpy)(tblock_norm);
	std::vector<double> vnorm = tblock_norm.expose();
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance / 2 of the normal
	double norm_mean = std::accumulate(vnorm.begin(), vnorm.end(), 0) / (double) vnorm.size();
	EXPECT_GT(norm_mean, mean - variance);
	EXPECT_LT(norm_mean, mean + variance);

	tensor<double> tscalar2(0);
	tensor<double> tblock2(shape);
	tensor<double> tblock_norm2(shape);
	tensor<double> ttransf2(std::vector<size_t>{(size_t) SUPERMARK});
	tfassign(ttransf2, empty_args);
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

	niassign(tblock_norm2);
	std::vector<double> vnorm2 = tblock_norm2.expose();
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance / 2 of the normal
	double norm_mean2 = std::accumulate(vnorm2.begin(), vnorm2.end(), 0) / (double) vnorm2.size();
	EXPECT_GT(norm_mean2, mean - variance);
	EXPECT_LT(norm_mean2, mean + variance);

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
	delete nicpy;
}


// cover transfer_func, const_init, rand_uniform
// move constructor and assignment
TEST(HANDLER, Move_C004)
{
	FUZZ::reset_logger();
	SUPERMARK = 0;
	transfer_func<double> tfassign(marked_forward);
	const_init<double> ciassign(0);
	rand_uniform<double> riassign(0, 1);
	rand_normal<double> niassign;

	SUPERMARK = FUZZ::getDouble(1, "SUPERMARK", {119, 221})[0];
	double scalar = FUZZ::getDouble(1, "scalar")[0];
	double low = FUZZ::getDouble(1, "low", {23, 127})[0];
	double high = FUZZ::getDouble(1, "high", {low*2, low*3+50})[0];
	double mean = FUZZ::getDouble(1, "mean", {-13, 23})[0];
	double variance = FUZZ::getDouble(1, "variance", {1, 32})[0];
	transfer_func<double> tf(marked_forward);
	const_init<double> ci(scalar);
	rand_uniform<double> ri(low, high);
	rand_normal<double> ni(mean, variance);

	transfer_func<double>* tfmv = tf.move();
	const_init<double>* cimv = ci.move();
	rand_uniform<double>* rimv = ri.move();
	rand_normal<double>* nimv = ni.move();

	tensorshape shape = random_def_shape();
	tensor<double> tscalar(0);
	tensor<double> tblock(shape);
	tensor<double> tblock_norm(shape);
	tensor<double> ttransf(std::vector<size_t>{(size_t) SUPERMARK});
	std::vector<const tensor<double>*> empty_args;
	(*tfmv)(ttransf, empty_args);
	std::vector<double> transfv = ttransf.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv.size());
	EXPECT_EQ(SUPERMARK, transfv[0]);

	(*cimv)(tscalar);
	EXPECT_EQ(scalar, tscalar.expose()[0]);

	(*rimv)(tblock);
	std::vector<double> v = tblock.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v[i]);
		EXPECT_GE(high, v[i]);
	}

	(*nimv)(tblock_norm);
	std::vector<double> vnorm = tblock_norm.expose();
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance / 2 of the normal
	double norm_mean = std::accumulate(vnorm.begin(), vnorm.end(), 0) / (double) vnorm.size();
	EXPECT_GT(norm_mean, mean - variance);
	EXPECT_LT(norm_mean, mean + variance);

	tfassign = std::move(*tfmv);
	ciassign = std::move(*cimv);
	riassign = std::move(*rimv);
	niassign = std::move(*nimv);

	tensor<double> tscalar2(0);
	tensor<double> tblock2(shape);
	tensor<double> tblock_norm2(shape);
	tensor<double> ttransf2(std::vector<size_t>{(size_t) SUPERMARK});
	tfassign(ttransf2, empty_args);
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

	niassign(tblock_norm2);
	std::vector<double> vnorm2 = tblock_norm2.expose();
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance of the normal
	double norm_mean2 = std::accumulate(vnorm2.begin(), vnorm2.end(), 0) / (double) vnorm2.size();
	EXPECT_GT(norm_mean2, mean - variance);
	EXPECT_LT(norm_mean2, mean + variance);

	delete tfmv;
	delete cimv;
	delete rimv;
	delete nimv;
}


#endif /* DISABLE_HANDLER_TEST */

#endif /* DISABLE_TENSOR_MODULE_TESTS */

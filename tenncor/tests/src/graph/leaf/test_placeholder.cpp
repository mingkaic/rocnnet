//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_connector.h"
#include "graph/leaf/placeholder.hpp"

#include "util_test.h"

#include "gtest/gtest.h"
#include "fuzz.h"


#ifndef DISABLE_PLACEHOLDER_TEST


TEST(PLACHOLDER, Constructor_G000)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape shape = random_def_shape();

	placeholder<double> place(shape, label1);
	std::vector<double> raw = FUZZ::getDouble(shape.n_elems());

	EXPECT_FALSE(place.good_status());
}


TEST(PLACHOLDER, Copy_G001)
{
	FUZZ::delim();
	std::vector<size_t> strns = FUZZ::getInt(2, {14, 29});
	std::string label1 = FUZZ::getString(strns[0]);
	std::string label2 = FUZZ::getString(strns[1]);
	tensorshape shape = random_def_shape();

	placeholder<double> assign(std::vector<size_t>{1});
	placeholder<double> place(shape, label1);
	std::vector<double> raw = FUZZ::getDouble(shape.n_elems());
	place = raw;
	placeholder<double>* pcpy = place.clone();
	assign = place;

	std::vector<double> cpyout = expose(pcpy);
	std::vector<double> assout = expose(&assign);

	size_t n = raw.size();
	ASSERT_EQ(cpyout.size(), n);
	ASSERT_EQ(assout.size(), n);
	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(raw[i], cpyout[i]);
		EXPECT_EQ(raw[i], assout[i]);
	}

	// check re-assignment after cloning
	std::vector<double> raw2 = FUZZ::getDouble(n);
	placeholder<double> assign2(std::vector<size_t>{1});
	placeholder<double> uninit(shape, label2);
	placeholder<double>* uninitcpy = uninit.clone();
	assign2 = uninit;

	// copy of uninitialized placeholders should still be able to initialize
	*pcpy = raw2;
	assign = raw2;
	// copy of initialized placeholders should be able to initialize
	*uninitcpy = raw2;
	assign2 = raw2;

	cpyout = expose(pcpy);
	assout = expose(&assign);
	std::vector<double> cpy2out = expose(uninitcpy);
	std::vector<double> ass2out = expose(&assign2);

	ASSERT_EQ(cpyout.size(), n);
	ASSERT_EQ(assout.size(), n);
	ASSERT_EQ(cpy2out.size(), n);
	ASSERT_EQ(ass2out.size(), n);
	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(raw2[i], cpyout[i]);
		EXPECT_EQ(raw2[i], assout[i]);
		EXPECT_EQ(raw2[i], cpy2out[i]);
		EXPECT_EQ(raw2[i], ass2out[i]);
	}

	delete pcpy;
	delete uninitcpy;
}


TEST(PLACHOLDER, Move_G001)
{
	FUZZ::delim();
	placeholder<double> assign(std::vector<size_t>{1});

	std::vector<size_t> strns = FUZZ::getInt(2, {14, 29});
	std::string label1 = FUZZ::getString(strns[0]);
	std::string label2 = FUZZ::getString(strns[1]);
	tensorshape shape = random_def_shape();

	placeholder<double> place(shape, label1);
	std::vector<double> raw = FUZZ::getDouble(shape.n_elems());

	size_t n = raw.size();
	place = raw;
	placeholder<double>* pmv = place.move();

	std::vector<double> mvout = expose(pmv);
	ASSERT_EQ(mvout.size(), n);

	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(raw[i], mvout[i]);
	}

	EXPECT_EQ(nullptr, place.get_eval());

	assign = std::move(*pmv);

	std::vector<double> assout = expose(&assign);
	ASSERT_EQ(assout.size(), n);

	EXPECT_EQ(nullptr, pmv->get_eval());

	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(raw[i], assout[i]);
	}

	delete pmv;
}


TEST(PLACHOLDER, AssignRaw_G002)
{
	mocker::usage_.clear();
	FUZZ::delim();
	std::vector<size_t> strns = FUZZ::getInt(3, {14, 29});
	std::string label1 = FUZZ::getString(strns[0]);
	std::string label2 = FUZZ::getString(strns[1]);
	std::string label3 = FUZZ::getString(strns[2]);
	tensorshape shape = random_def_shape();
	tensorshape part = make_partial(shape.as_list());

	placeholder<double> place(shape, label1);
	placeholder<double> place2(part, label2);
	std::vector<double> raw = FUZZ::getDouble(shape.n_elems());

	mock_connector conn({&place}, label3);
	conn.inst_ = "conn";

	EXPECT_FALSE(place.good_status());
	place = raw;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::update1", 1));

	EXPECT_TRUE(place.good_status());
	const tensor<double>* placer = place.get_eval();
	EXPECT_TRUE(placer->is_alloc());
	EXPECT_TRUE(tensorshape_equal(shape, placer->get_shape()));
	std::vector<double> out = placer->expose();
	for (size_t i = 0, n = out.size(); i < n; i++)
	{
		EXPECT_EQ(raw[i], out[i]);
	}

	// partial placeholders may guess shape for input vector
	// place2 will succeed since place2 is made partial from initial shape
	place2 = raw;
	EXPECT_TRUE(place2.good_status());
	const tensor<double>* placer2 = place2.get_eval();
	EXPECT_TRUE(placer2->is_alloc());
	out = placer2->expose();
	for (size_t i = 0, n = out.size(); i < n; i++)
	{
		EXPECT_EQ(raw[i], out[i]);
	}
}


TEST(PLACHOLDER, AssignTensor_G003)
{
	mocker::usage_.clear();
	FUZZ::delim();
	std::vector<size_t> strns = FUZZ::getInt(2, {14, 29});
	std::string label1 = FUZZ::getString(strns[0]);
	std::string label2 = FUZZ::getString(strns[1]);
	tensorshape shape = random_def_shape();

	placeholder<double> place(shape, label1);

	double c = FUZZ::getDouble(1)[0];
	const_init<double> cinit(c);
	tensor<double> rawtens(shape);
	cinit(rawtens);

	mock_connector conn({&place}, label2);
	conn.inst_ = "conn";

	EXPECT_FALSE(place.good_status());
	place = rawtens;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::update1", 1));
	EXPECT_FALSE(rawtens.is_alloc());

	EXPECT_TRUE(place.good_status());
	const tensor<double>* placer = place.get_eval();
	EXPECT_TRUE(placer->is_alloc());
	EXPECT_TRUE(tensorshape_equal(shape, placer->get_shape()));
	std::vector<double> out = placer->expose();
	for (size_t i = 0, n = out.size(); i < n; i++)
	{
		EXPECT_EQ(c, out[i]);
	}
}


TEST(PLACHOLDER, GetLeaf_G004)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape shape = random_def_shape();

	placeholder<double> place(shape, label1);

	inode<double>* zaro = place.get_leaf(nullptr);
	EXPECT_TRUE(*zaro == 0.0);
}


TEST(PLACHOLDER, GetLeaves_G005)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape shape = random_def_shape();

	placeholder<double> place(shape, label1);

	typename inode<double>::GRAD_CACHE leafset;
	place.get_leaves(leafset);
	EXPECT_TRUE(leafset.empty());
}


#endif /* DISABLE_PLACEHOLDER_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */

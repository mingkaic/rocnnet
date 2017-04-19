//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_leaf.h"


//#define DISABLE_LEAF_TEST
#ifndef DISABLE_LEAF_TEST


// covers ileaf
// copy constructor and assignment
TEST(LEAF, Copy_C000)
{
	FUZZ::delim();
	mock_leaf assign("");
	mock_leaf assign2("");

	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape comp = random_def_shape();
	tensorshape part = make_partial(comp.as_list());
	mock_leaf res(comp, label1);
	mock_leaf res2(part, label2);
	res.set_good();

	bool initstatus = res.good_status();
	bool initstatus2 = res2.good_status();
	const tensor<double>* init_data = res.get_eval();
	const tensor<double>* init_data2 = res2.get_eval(); // res2 is not good

	ileaf<double>* cpy = res.clone();
	ileaf<double>* cpy2 = res2.clone();
	assign = res;
	assign2 = res2;

	bool cpystatus = cpy->good_status();
	bool cpystatus2 = cpy2->good_status();
	const tensor<double>* cpy_data = cpy->get_eval();
	const tensor<double>* cpy_data2 = cpy2->get_eval();

	bool assignstatus = assign.good_status();
	bool assignstatus2 = assign2.good_status();
	const tensor<double>* assign_data = assign.get_eval();
	const tensor<double>* assign_data2 = assign2.get_eval();

	EXPECT_EQ(initstatus, cpystatus);
	EXPECT_EQ(initstatus, assignstatus);
	EXPECT_EQ(initstatus2, cpystatus2);
	EXPECT_EQ(initstatus2, assignstatus2);
	EXPECT_NE(initstatus, cpystatus2);
	EXPECT_NE(initstatus, assignstatus2);
	EXPECT_NE(initstatus2, cpystatus);
	EXPECT_NE(initstatus2, assignstatus);

	// expect deep copy
	EXPECT_NE(init_data, cpy_data);
	EXPECT_NE(init_data, assign_data);
	EXPECT_EQ(nullptr, init_data2);
	EXPECT_EQ(nullptr, cpy_data2);
	EXPECT_EQ(nullptr, assign_data2);
	ASSERT_NE(nullptr, init_data);
	ASSERT_NE(nullptr, cpy_data);
	ASSERT_NE(nullptr, assign_data);
	EXPECT_TRUE(tensorshape_equal(init_data->get_shape(), cpy_data->get_shape()));
	EXPECT_TRUE(tensorshape_equal(init_data->get_shape(), assign_data->get_shape()));
	// we're checking tensor copy over
	// data isn't initialized at this point, so we're exposing garabage.
	// regardless, deep copy would still copy over memory content since
	// [IMPORTANT!] tensor has no initialization data and we just pretended that status is good
	std::vector<double> idata = init_data->expose();
	std::vector<double> cdata = cpy_data->expose();
	std::vector<double> adata = assign_data->expose();

	EXPECT_TRUE(std::equal(idata.begin(), idata.end(), cdata.begin()));
	EXPECT_TRUE(std::equal(idata.begin(), idata.end(), adata.begin()));

	delete cpy;
	delete cpy2;
}


// covers ileaf
// move constructor and assignment
TEST(LEAF, Move_C000)
{
	FUZZ::delim();
	mock_leaf assign("");
	mock_leaf assign2("");

	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape comp = random_def_shape();
	tensorshape part = make_partial(comp.as_list());
	mock_leaf res(comp, label1);
	mock_leaf res2(part, label2);
	res.set_good();

	bool initstatus = res.good_status();
	bool initstatus2 = res2.good_status();
	const tensor<double>* init_data = res.get_eval();
	const tensor<double>* init_data2 = res2.get_eval();

	ileaf<double>* mv = res.move();
	ileaf<double>* mv2 = res2.move();

	bool mvstatus = mv->good_status();
	bool mvstatus2 = mv2->good_status();
	// ensure shallow copy
	const tensor<double>* mv_data = mv->get_eval();
	const tensor<double>* mv_data2 = mv2->get_eval();
	EXPECT_EQ(init_data, mv_data);
	EXPECT_EQ(init_data2, mv_data2);
	EXPECT_NE(init_data2, mv_data);
	EXPECT_NE(init_data, mv_data2);
	EXPECT_EQ(nullptr, res.get_eval());
	EXPECT_EQ(nullptr, res2.get_eval());

	mock_leaf* mmv = static_cast<mock_leaf*>(mv);
	mock_leaf* mmv2 = static_cast<mock_leaf*>(mv2);
	assign = std::move(*mmv);
	assign2 = std::move(*mmv2);

	bool assignstatus = assign.good_status();
	bool assignstatus2 = assign2.good_status();
	// ensure shallow copy
	const tensor<double>* assign_data = assign.get_eval();
	const tensor<double>* assign_data2 = assign2.get_eval();
	EXPECT_EQ(mv_data, assign_data);
	EXPECT_EQ(mv_data2, assign_data2);
	EXPECT_NE(mv_data, assign_data2);
	EXPECT_NE(mv_data2, assign_data);
	EXPECT_EQ(nullptr, mv->get_eval());
	EXPECT_EQ(nullptr, mv2->get_eval());

	EXPECT_EQ(initstatus, mvstatus);
	EXPECT_EQ(initstatus, assignstatus);
	EXPECT_EQ(initstatus2, mvstatus2);
	EXPECT_EQ(initstatus2, assignstatus2);
	EXPECT_NE(initstatus, mvstatus2);
	EXPECT_NE(initstatus, assignstatus2);
	EXPECT_NE(initstatus2, mvstatus);
	EXPECT_NE(initstatus2, assignstatus);

	delete mv;
	delete mv2;
}


// covers ileaf get_shape
TEST(LEAF, GetShape_C001)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape shape1 = random_shape();
	tensorshape shape2 = random_shape();
	mock_leaf res(shape1, label1);
	mock_leaf res2(shape2, label2);

	tensorshape rshape1 = res.get_shape();
	tensorshape rshape2 = res2.get_shape();
	EXPECT_TRUE(tensorshape_equal(shape1, rshape1));
	EXPECT_TRUE(tensorshape_equal(shape2, rshape2));

	EXPECT_EQ(tensorshape_equal(shape1, shape2),
		tensorshape_equal(shape2, rshape1));
	EXPECT_EQ(tensorshape_equal(shape1, shape2),
		tensorshape_equal(shape1, rshape2));
}


// covers ileaf get_eval
TEST(LEAF, GetEval_C002)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape comp = random_def_shape();
	tensorshape part = make_partial(comp.as_list());
	mock_leaf res(comp, label1);
	mock_leaf res2(part, label2);
	// pretend they're both good (initialized)
	res.set_good();
	res2.set_good();

	const tensor<double>* rout = res.get_eval();
	const tensor<double>* r2out = res2.get_eval();

	ASSERT_NE(nullptr, rout);
	ASSERT_NE(nullptr, r2out);
	EXPECT_TRUE(tensorshape_equal(comp, rout->get_shape()));
	EXPECT_TRUE(tensorshape_equal(part, r2out->get_shape()));
	EXPECT_TRUE(rout->is_alloc());
	EXPECT_FALSE(r2out->is_alloc());
}


// covers ileaf good_status
TEST(LEAF, GoodStatus_C003)
{
	FUZZ::delim();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
	tensorshape comp = random_def_shape();
	tensorshape part = make_partial(comp.as_list());
	mock_leaf res(comp, label1);
	mock_leaf res2(part, label2);
	res.set_good();

	EXPECT_TRUE(res.good_status());
	EXPECT_FALSE(res2.good_status());
}


#endif /* DISABLE_LEAF_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */

//
// Created by Mingkai Chen on 2016-08-29.
//

#include "tensor_test_util.h"


// undefined
static const nnet::tensorshape incom_ts;
// partially defined
static const nnet::tensorshape pcom_ts(std::vector<size_t>{0, 1, 2});
// fully defined
static const nnet::tensorshape com_ts(std::vector<size_t>{1, 2, 3});
static const nnet::tensorshape com2(std::vector<size_t>{4, 5});
// additional partials (that are compatible with full)
static const nnet::tensorshape pcom1(std::vector<size_t>{0, 2, 3});
static const nnet::tensorshape pcom2(std::vector<size_t>{0, 0, 3});
// guaranteed to be not compatible
static const nnet::tensorshape bad_by_value(std::vector<size_t>{10, 10, 10});
static const nnet::tensorshape bad_by_rank(std::vector<size_t>{2, 2, 2, 2});


TEST(TENSORSHAPE, Copy)
{
	nnet::tensorshape local_incom(incom_ts);
	nnet::tensorshape local_pcom(pcom_ts);
	nnet::tensorshape local_com(com_ts);
	nnet::tensorshape assign_incom;
	nnet::tensorshape assign_pcom;
	nnet::tensorshape assign_com;
	assign_incom = incom_ts;
	assign_pcom = pcom_ts;
	assign_com = com_ts;
	ASSERT_TRUE(tensorshape_equal(local_incom, incom_ts));
	ASSERT_TRUE(tensorshape_equal(local_pcom, pcom_ts));
	ASSERT_TRUE(tensorshape_equal(local_com, com_ts));
	ASSERT_TRUE(tensorshape_equal(assign_incom, incom_ts));
	ASSERT_TRUE(tensorshape_equal(assign_pcom, pcom_ts));
	ASSERT_TRUE(tensorshape_equal(assign_com, com_ts));
}


// Behavior A000
TEST(TENSORSHAPE, Compatible_A000)
{
	// undefined are compatible with everything
	std::vector<const nnet::tensorshape*> def_shapes =
		{&pcom_ts, &com_ts, &pcom1, &pcom2, &incom_ts, &bad_by_value, &bad_by_rank};
	for (const nnet::tensorshape* shape : def_shapes)
	{
		EXPECT_TRUE(incom_ts.is_compatible_with(*shape));
	}
	
	// partial 1 and 2 are compatible with full but not bads
	EXPECT_TRUE(pcom1.is_compatible_with(com_ts));
	EXPECT_TRUE(pcom2.is_compatible_with(com_ts));
	EXPECT_FALSE(pcom_ts.is_compatible_with(com_ts));

	std::vector<const nnet::tensorshape*> def_shapes2 =
		{&pcom_ts, &com_ts, &pcom1, &pcom2};
	// bads are expected to be incompatible with full and partials
	for (const nnet::tensorshape* shape : def_shapes2)
	{
		EXPECT_FALSE(bad_by_value.is_compatible_with(*shape));
		EXPECT_FALSE(bad_by_rank.is_compatible_with(*shape));
	}
}


// Behavior A001
TEST(TENSORSHAPE, MergeComp_A001)
{
	std::vector<const nnet::tensorshape*> def_shapes =
		{&pcom_ts, &com_ts, &pcom1, &pcom2, &incom_ts, &bad_by_value, &bad_by_rank};
	// incomplete shape can merge with anything
	for (const nnet::tensorshape* shape : def_shapes)
	{
		nnet::tensorshape merged = incom_ts.merge_with(*shape);
		// we're expecting merged shape to be the same as input shape
		EXPECT_TRUE(tensorshape_equal(merged, *shape));
	}

	// merge fully known shapes
	std::vector<const nnet::tensorshape*> def_shapes2 =
		{&pcom1, &pcom2};

	for (const nnet::tensorshape* shape : def_shapes2)
	{
		nnet::tensorshape full_merge = com_ts.merge_with(*shape);
		// we're expecting merged shape to be the same as fully defined shape
		EXPECT_TRUE(tensorshape_equal(full_merge, com_ts));
	}
}


// Behavior A002
TEST(TENSORSHAPE, MergeIncomp_A001)
{
	std::vector<const nnet::tensorshape*> good_shapes =
		{&pcom_ts, &com_ts, &pcom1, &pcom2};
	
	for (const nnet::tensorshape* shape : good_shapes)
	{
		EXPECT_THROW({ bad_by_value.merge_with(*shape); }, 
			std::logic_error);
		EXPECT_THROW({ bad_by_rank.merge_with(*shape); }, 
			std::logic_error);
	}
}


// Behavior A003
TEST(TENSORSHAPE, Concat_A003)
{
	nnet::tensorshape none1 = incom_ts.concatenate(com_ts);
	EXPECT_FALSE(none1.is_part_defined());
	nnet::tensorshape none2 = com_ts.concatenate(incom_ts);
	EXPECT_FALSE(none2.is_part_defined());

	// 1, 2, 3, 4, 5
	nnet::tensorshape straight = com_ts.concatenate(com2);
	// 4, 5, 1, 2, 3
	nnet::tensorshape backcat = com2.concatenate(com_ts);

	nnet::tensorshape expected_str8 = std::vector<size_t>{1, 2, 3, 4, 5};
	nnet::tensorshape expected_back = std::vector<size_t>{4, 5, 1, 2, 3};
	EXPECT_TRUE(straight.is_compatible_with(expected_str8));
	EXPECT_TRUE(backcat.is_compatible_with(expected_back));
}


// Behavior A004
TEST(TENSORSHAPE, PartiallyDefined_A004)
{
	EXPECT_FALSE(incom_ts.is_part_defined());
	EXPECT_TRUE(pcom_ts.is_part_defined());
	EXPECT_TRUE(com_ts.is_part_defined());
}


// Behavior A005
TEST(TENSORSHAPE, FullyDefine_A005)
{
	EXPECT_FALSE(incom_ts.is_fully_defined());
	EXPECT_FALSE(pcom_ts.is_fully_defined());
	EXPECT_TRUE(com_ts.is_fully_defined());
}


// UTILITIES
TEST(TENSORSHAPE, Assertions)
{
	// rank assertion
	incom_ts.assert_has_rank(10);
	com_ts.assert_has_rank(3);
	ASSERT_DEATH({ com_ts.assert_has_rank(1); }, ".*");
	// rank equality assertion
	incom_ts.assert_same_rank(pcom_ts);
	bad_by_value.assert_same_rank(pcom_ts);
	ASSERT_DEATH({ bad_by_rank.assert_same_rank(pcom_ts); }, ".*");
	// is fully defined
	com_ts.assert_is_fully_defined();
	ASSERT_DEATH({ pcom_ts.assert_is_fully_defined(); }, ".*");
	ASSERT_DEATH({ incom_ts.assert_is_fully_defined(); }, ".*");
}


TEST(TENSORSHAPE, Accessors)
{
	// rank
	EXPECT_EQ(0, incom_ts.n_dims());
	EXPECT_EQ(3, com_ts.n_dims());
	// dimension vector
	std::vector<nnet::dimension> full_out = com_ts.dims();
	size_t d_size = full_out.size();
	ASSERT_EQ(3, d_size);
	for (int i = 0; i < d_size; i++)
	{
		EXPECT_EQ(i+1, full_out[i].value_);
	}
	ASSERT_EQ(incom_ts.dims().size(), 0);
	// dimension vector as integers
	std::vector<size_t> vout = pcom_ts.as_list();
	size_t l_size = vout.size();
	ASSERT_EQ(3, l_size);
	for (int i = 0; i < l_size; i++) {
		EXPECT_EQ(i, vout[i]);
	}
	ASSERT_EQ(incom_ts.as_list().size(), 0);
}


// RANK CREATION
TEST(TENSORSHAPE, RankCreation)
{
	nnet::tensorshape ts1_cpy = com_ts.with_rank(3);
	nnet::tensorshape ts2_cpy = com2.with_rank(2);
	EXPECT_TRUE(tensorshape_equal(com_ts, ts1_cpy));
	EXPECT_TRUE(tensorshape_equal(com2, ts2_cpy));
	
	ts1_cpy = com_ts.with_rank_at_least(3);
	ts2_cpy = com2.with_rank_at_least(2);
	nnet::tensorshape ts1_cpy2 = com_ts.with_rank_at_least(2);
	EXPECT_TRUE(tensorshape_equal(com_ts, ts1_cpy));
	EXPECT_TRUE(tensorshape_equal(com_ts, ts1_cpy2));
	EXPECT_TRUE(tensorshape_equal(com2, ts2_cpy));
	
	ts1_cpy = com_ts.with_rank_at_most(3);
	ts2_cpy = com2.with_rank_at_most(2);
	ts1_cpy2 = com_ts.with_rank_at_most(4);
	EXPECT_TRUE(tensorshape_equal(com_ts, ts1_cpy));
	EXPECT_TRUE(tensorshape_equal(com_ts, ts1_cpy2));
	EXPECT_TRUE(tensorshape_equal(com2, ts2_cpy));
	
	// with_rank_*(i) on undefined shapes generate partially defined shapes of rank i
	for (size_t i = 0; i < 3; i++)
	{
		nnet::tensorshape tsi = incom_ts.with_rank(i);
		nnet::tensorshape tsi2 = incom_ts.with_rank_at_least(i);
		nnet::tensorshape tsi3 = incom_ts.with_rank_at_most(i);
		std::vector<size_t> zeros = tsi.as_list();
		EXPECT_EQ(zeros.size(), i);
		for (size_t j : zeros)
		{
			EXPECT_EQ(j, 0);
		}
		EXPECT_TRUE(tensorshape_equal(tsi, tsi2));
		EXPECT_TRUE(tensorshape_equal(tsi, tsi3));
	}

	EXPECT_THROW({ com_ts.with_rank(2); }, std::logic_error);
	EXPECT_THROW({ com_ts.with_rank(4); }, std::logic_error);
	EXPECT_THROW({ com_ts.with_rank_at_least(4); }, std::logic_error);
	EXPECT_THROW({ com_ts.with_rank_at_most(2); }, std::logic_error);
}


// TENSORSHAPE TO PROTOCOL BUFFER
TEST(TENSORSHAPE, as_proto) {}

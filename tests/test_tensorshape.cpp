//
//  test_dimension.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../include/graph.hpp"

static bool tensor_shape_equal (
	nnet::tensor_shape const & ts1,
	nnet::tensor_shape const & ts2) {
	if (ts1.is_part_defined() && ts2.is_part_defined()) {
		std::vector<size_t> dims1 = ts1.as_list();
		std::vector<size_t> dims2 = ts2.as_list();
		return std::equal(dims1.begin(), dims1.end(), dims2.begin());
	} else if (ts1.is_part_defined() == ts2.is_part_defined()) {
		return true;
	}
	return false;
}

TEST(DIMENSION, assert_is_compatible_with) {
	nnet::dimension d0(0); // unknown
	nnet::dimension d1(1);
	nnet::dimension d2(2);

	d0.assert_is_compatible_with(d0);
	d0.assert_is_compatible_with(d1);
	d0.assert_is_compatible_with(d2);

	d1.assert_is_compatible_with(d0);
	d1.assert_is_compatible_with(d1);
	EXPECT_DEATH({ d1.assert_is_compatible_with(d2); }, ".*");

	d2.assert_is_compatible_with(d0);
	d2.assert_is_compatible_with(d2);
	ASSERT_DEATH({ d2.assert_is_compatible_with(d1); }, ".*");
}


TEST(DIMENSION, is_compatible_with) {
	nnet::dimension d0(0); // unknown
	nnet::dimension d1(1);
	nnet::dimension d2(2);

	EXPECT_TRUE(d0.is_compatible_with(d0));
	EXPECT_TRUE(d0.is_compatible_with(d1));
	EXPECT_TRUE(d0.is_compatible_with(d2));

	EXPECT_TRUE(d1.is_compatible_with(d0));
	EXPECT_TRUE(d1.is_compatible_with(d1));
	EXPECT_FALSE(d1.is_compatible_with(d2));

	EXPECT_TRUE(d2.is_compatible_with(d0));
	EXPECT_FALSE(d2.is_compatible_with(d1));
	ASSERT_TRUE(d2.is_compatible_with(d2));
}


TEST(DIMENSION, merge_with) {
	nnet::dimension d0(0); // unknown
	nnet::dimension d1(1);
	nnet::dimension d2(2);

	nnet::dimension d00 = d0.merge_with(d0);
	nnet::dimension d01 = d0.merge_with(d1);
	nnet::dimension d02 = d0.merge_with(d2);
	EXPECT_EQ(d00.value, d0.value);
	EXPECT_EQ(d01.value, d1.value);
	EXPECT_EQ(d02.value, d2.value);

	nnet::dimension d10 = d1.merge_with(d0);
	nnet::dimension d11 = d1.merge_with(d1);
	EXPECT_EQ(d10.value, d1.value);
	EXPECT_EQ(d11.value, d1.value);
	EXPECT_THROW({ d1.merge_with(d2); }, std::logic_error);

	nnet::dimension d20 = d2.merge_with(d0);
	nnet::dimension d22 = d2.merge_with(d2);
	EXPECT_EQ(d20.value, d2.value);
	EXPECT_EQ(d22.value, d2.value);
	ASSERT_THROW({ d2.merge_with(d1); }, std::logic_error);
}


TEST(TENSORSHAPE, is_compatible_with) {
	std::vector<size_t> part1 = {0, 1, 2};
	std::vector<size_t> part2 = {3, 1, 0};
	std::vector<size_t> part3 = {3, 0, 0};
	std::vector<size_t> bad = {2, 0, 0};
	std::vector<size_t> com = {3, 1, 2};
	nnet::tensor_shape incom_ts;
	nnet::tensor_shape com_ts1(com);
	nnet::tensor_shape com_ts2(com);
	nnet::tensor_shape p_ts1(part1);
	nnet::tensor_shape p_ts2(part2);
	nnet::tensor_shape p_ts3(part3);
	nnet::tensor_shape bad_ts(bad);
	EXPECT_TRUE(incom_ts.is_compatible_with(com_ts1));
	EXPECT_TRUE(com_ts1.is_compatible_with(com_ts2));

	EXPECT_TRUE(p_ts1.is_compatible_with(p_ts2));
	EXPECT_TRUE(p_ts2.is_compatible_with(p_ts3));
	EXPECT_TRUE(p_ts3.is_compatible_with(p_ts1));

	EXPECT_TRUE(p_ts1.is_compatible_with(com_ts1));
	EXPECT_TRUE(p_ts2.is_compatible_with(com_ts1));
	EXPECT_TRUE(p_ts3.is_compatible_with(com_ts1));

	ASSERT_FALSE(bad_ts.is_compatible_with(com_ts1));
}


TEST(TENSORSHAPE, is_part_defined) {
	std::vector<size_t> v = {0, 1, 2};
	nnet::tensor_shape incom_ts;
	nnet::tensor_shape com_ts(v);
	EXPECT_FALSE(incom_ts.is_part_defined());
	ASSERT_TRUE(com_ts.is_part_defined());
}


TEST(TENSORSHAPE, is_fully_defined) {
	std::vector<size_t> v = {0, 1, 2};
	std::vector<size_t> fv = {3, 1, 2};
	nnet::tensor_shape incom_ts;
	nnet::tensor_shape pcom_ts(v);
	nnet::tensor_shape fcom_ts(fv);
	EXPECT_FALSE(incom_ts.is_fully_defined());
	EXPECT_FALSE(pcom_ts.is_fully_defined());
	EXPECT_TRUE(fcom_ts.is_fully_defined());
}


TEST(TENSORSHAPE, assert_has_rank) {
	std::vector<size_t> v = {0, 1, 2};
	nnet::tensor_shape incom_ts;
	nnet::tensor_shape com_ts(v);
	incom_ts.assert_has_rank(10);
	com_ts.assert_has_rank(3);

	ASSERT_DEATH({ com_ts.assert_has_rank(1); }, ".*");
}


TEST(TENSORSHAPE, assert_same_rank) {
	std::vector<size_t> v1 = {0, 1, 2};
	std::vector<size_t> v2 = {3, 4, 5};
	std::vector<size_t> v3 = {3, 4};
	nnet::tensor_shape incom_ts;
	nnet::tensor_shape ts1(v1);
	nnet::tensor_shape ts2(v2);
	nnet::tensor_shape ts3(v3);

	incom_ts.assert_same_rank(ts1);
	ts1.assert_same_rank(incom_ts);
	ts1.assert_same_rank(ts2);
	ts2.assert_same_rank(ts1);

	ASSERT_DEATH({ ts2.assert_same_rank(ts3); }, ".*");
}


TEST(TENSORSHAPE, assert_is_fully_defined) {
	std::vector<size_t> v = {0, 1, 2};
	std::vector<size_t> fv = {3, 1, 2};
	nnet::tensor_shape pcom_ts(v);
	nnet::tensor_shape fcom_ts(fv);
	nnet::tensor_shape incom_ts;
	fcom_ts.assert_is_fully_defined();
	ASSERT_DEATH({ pcom_ts.assert_is_fully_defined(); }, ".*");
	ASSERT_DEATH({ incom_ts.assert_is_fully_defined(); }, ".*");
}


TEST(TENSORSHAPE, n_dims) {
	std::vector<size_t> v = {0, 1, 2};
	nnet::tensor_shape ts(v);
	nnet::tensor_shape incom_ts;

	EXPECT_EQ(incom_ts.n_dims(), 0);
	ASSERT_EQ(ts.n_dims(), v.size());
}


TEST(TENSORSHAPE, dims) {
	std::vector<size_t> v = {0, 1, 2};
	nnet::tensor_shape ts(v);
	nnet::tensor_shape incom_ts;

	std::vector<nnet::dimension> vout = ts.dims();
	ASSERT_EQ(v.size(), vout.size());
	for (int i = 0; i < v.size(); i++) {
		EXPECT_EQ(v[i], vout[i].value);
	}
	ASSERT_EQ(incom_ts.dims().size(), 0);
}


TEST(TENSORSHAPE, as_list) {
	std::vector<size_t> v = {0, 1, 2};
	nnet::tensor_shape ts(v);
	nnet::tensor_shape incom_ts;

	std::vector<size_t> vout = ts.as_list();
	ASSERT_EQ(v.size(), vout.size());
	for (int i = 0; i < v.size(); i++) {
		EXPECT_EQ(v[i], vout[i]);
	}
	ASSERT_EQ(incom_ts.as_list().size(), 0);
}


TEST(TENSORSHAPE, merge_with) {
	std::vector<size_t> bad = {0};
	std::vector<size_t> bad2 = {10, 10, 10};
	std::vector<size_t> part = {0, 1, 2};
	std::vector<size_t> full = {3, 1, 2};
	nnet::tensor_shape incom_ts;
	nnet::tensor_shape bad_ts(bad);
	nnet::tensor_shape bad_ts2(bad2);
	nnet::tensor_shape part_ts(part);
	nnet::tensor_shape full_ts(full);

	// bad tensorshape of rank 1
	std::vector<nnet::tensor_shape*> tss = {&part_ts, &full_ts};
	for (nnet::tensor_shape* ts : tss) {
		EXPECT_THROW({ bad_ts.merge_with(*ts); }, std::logic_error);
	}

	// bad tensorshape of rank 3
	EXPECT_THROW({ bad_ts2.merge_with(full_ts); }, std::logic_error);

	// incomplete shape test
	for (nnet::tensor_shape* ts : tss) {
		nnet::tensor_shape ts_cpy = incom_ts.merge_with(*ts);
		EXPECT_TRUE(tensor_shape_equal(*ts, ts_cpy));
	}

	// fully known shape test
	for (nnet::tensor_shape* ts : tss) {
		nnet::tensor_shape full_cpy = full_ts.merge_with(*ts);
		EXPECT_TRUE(tensor_shape_equal(full_ts, full_cpy));
	}

	// partially known test
	nnet::tensor_shape part_cpy = part_ts.merge_with(incom_ts);
	EXPECT_TRUE(tensor_shape_equal(part_ts, part_cpy));
	nnet::tensor_shape full_cpy = part_ts.merge_with(full_ts);
	ASSERT_TRUE(tensor_shape_equal(full_ts, full_cpy));
}


TEST(TENSORSHAPE, concatenate) {
	nnet::tensor_shape incom_ts;
	std::vector<size_t> v1 = {1, 2, 3};
	std::vector<size_t> v2 = {4, 5};
	nnet::tensor_shape ts1(v1);
	nnet::tensor_shape ts2(v2);

	nnet::tensor_shape tnone1 = incom_ts.concatenate(ts1);
	EXPECT_TRUE(tensor_shape_equal(incom_ts, tnone1));
	nnet::tensor_shape tnone2 = ts1.concatenate(incom_ts);
	EXPECT_TRUE(tensor_shape_equal(incom_ts, tnone2));

	nnet::tensor_shape straight = ts1.concatenate(ts2);
	nnet::tensor_shape backcat = ts2.concatenate(ts1);

	std::vector<size_t> d1 = straight.as_list();
	std::vector<size_t> d2 = backcat.as_list();

	ASSERT_EQ(d1.size(), v1.size()+v2.size());
	ASSERT_EQ(d2.size(), v1.size()+v2.size());

	size_t i = 1;
	for (size_t d : d1) {
		EXPECT_EQ(d, i++);
	}
	for (i = 0; i < v2.size(); i++) {
		EXPECT_EQ(d2[i], v2[i]);
	}
	for (size_t j = 0; j < v1.size(); j++) {
		EXPECT_EQ(d2[i+j], v1[j]);
	}
}


TEST(TENSORSHAPE, with_rank) {
	nnet::tensor_shape incom_ts;
	std::vector<size_t> v1 = {1, 2, 3};
	std::vector<size_t> v2 = {4, 5};
	nnet::tensor_shape ts1(v1);
	nnet::tensor_shape ts2(v2);

	nnet::tensor_shape ts1_cpy = ts1.with_rank(3);
	nnet::tensor_shape ts2_cpy = ts2.with_rank(2);
	EXPECT_TRUE(tensor_shape_equal(ts1, ts1_cpy));
	EXPECT_TRUE(tensor_shape_equal(ts2, ts2_cpy));

	EXPECT_THROW({ ts1.with_rank(2); }, std::logic_error);

	EXPECT_THROW({ ts1.with_rank(4); }, std::logic_error);

	for (size_t i = 0; i < 3; i++) {
		nnet::tensor_shape tsi = incom_ts.with_rank(i);
		std::vector<size_t> zeros = tsi.as_list();
		EXPECT_EQ(zeros.size(), i);
		for (size_t j : zeros) {
			EXPECT_EQ(j, 0);
		}
	}
}


TEST(TENSORSHAPE, with_rank_at_least) {
	nnet::tensor_shape incom_ts;
	std::vector<size_t> v1 = {1, 2, 3};
	std::vector<size_t> v2 = {4, 5};
	nnet::tensor_shape ts1(v1);
	nnet::tensor_shape ts2(v2);

	nnet::tensor_shape ts1_cpy = ts1.with_rank_at_least(3);
	nnet::tensor_shape ts2_cpy = ts2.with_rank_at_least(2);
	nnet::tensor_shape ts1_cpy2 = ts1.with_rank_at_least(2);
	EXPECT_TRUE(tensor_shape_equal(ts1, ts1_cpy));
	EXPECT_TRUE(tensor_shape_equal(ts2, ts2_cpy));
	EXPECT_TRUE(tensor_shape_equal(ts1, ts1_cpy2));

	EXPECT_THROW({ ts1.with_rank_at_least(4); }, std::logic_error);

	for (size_t i = 0; i < 3; i++) {
		nnet::tensor_shape tsi = incom_ts.with_rank_at_least(i);
		std::vector<size_t> zeros = tsi.as_list();
		EXPECT_EQ(zeros.size(), i);
		for (size_t j : zeros) {
			EXPECT_EQ(j, 0);
		}
	}
}


TEST(TENSORSHAPE, with_rank_at_most) {
	nnet::tensor_shape incom_ts;
	std::vector<size_t> v1 = {1, 2, 3};
	std::vector<size_t> v2 = {4, 5};
	nnet::tensor_shape ts1(v1);
	nnet::tensor_shape ts2(v2);

	nnet::tensor_shape ts1_cpy = ts1.with_rank_at_most(3);
	nnet::tensor_shape ts2_cpy = ts2.with_rank_at_most(2);
	nnet::tensor_shape ts1_cpy2 = ts1.with_rank_at_most(4);
	EXPECT_TRUE(tensor_shape_equal(ts1, ts1_cpy));
	EXPECT_TRUE(tensor_shape_equal(ts2, ts2_cpy));
	EXPECT_TRUE(tensor_shape_equal(ts1, ts1_cpy2));

	EXPECT_THROW({ ts1.with_rank_at_most(2); }, std::logic_error);

	for (size_t i = 0; i < 3; i++) {
		nnet::tensor_shape tsi = incom_ts.with_rank_at_most(i);
		std::vector<size_t> zeros = tsi.as_list();
		EXPECT_EQ(zeros.size(), i);
		for (size_t j : zeros) {
			EXPECT_EQ(j, 0);
		}
	}
}


TEST(TENSORSHAPE, as_proto) {}

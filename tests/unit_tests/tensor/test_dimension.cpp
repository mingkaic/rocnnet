//
// Created by Mingkai Chen on 2016-11-17.
//

#include "gtest/gtest.h"
#include "tensor/tensorshape.hpp"


TEST(DIMENSION, Compatible) {
	nnet::dimension d0(0); // unknown
	nnet::dimension d1(1);
	nnet::dimension d2(2);

	// assertion
	d0.assert_is_compatible_with(d0);
	d0.assert_is_compatible_with(d1);
	d0.assert_is_compatible_with(d2);
	d1.assert_is_compatible_with(d0);
	d1.assert_is_compatible_with(d1);
	EXPECT_DEATH({ d1.assert_is_compatible_with(d2); }, ".*");
	d2.assert_is_compatible_with(d0);
	d2.assert_is_compatible_with(d2);
	ASSERT_DEATH({ d2.assert_is_compatible_with(d1); }, ".*");

	// check
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


TEST(DIMENSION, Merge) {
	nnet::dimension d0(0); // unknown
	nnet::dimension d1(1);
	nnet::dimension d2(2);

	nnet::dimension d00 = d0.merge_with(d0);
	nnet::dimension d01 = d0.merge_with(d1);
	nnet::dimension d02 = d0.merge_with(d2);
	EXPECT_EQ(d00.value_, d0.value_);
	EXPECT_EQ(d01.value_, d1.value_);
	EXPECT_EQ(d02.value_, d2.value_);

	nnet::dimension d10 = d1.merge_with(d0);
	nnet::dimension d11 = d1.merge_with(d1);
	EXPECT_EQ(d10.value_, d1.value_);
	EXPECT_EQ(d11.value_, d1.value_);
	EXPECT_THROW({ d1.merge_with(d2); }, std::logic_error);

	nnet::dimension d20 = d2.merge_with(d0);
	nnet::dimension d22 = d2.merge_with(d2);
	EXPECT_EQ(d20.value_, d2.value_);
	EXPECT_EQ(d22.value_, d2.value_);
	ASSERT_THROW({ d2.merge_with(d1); }, std::logic_error);
}

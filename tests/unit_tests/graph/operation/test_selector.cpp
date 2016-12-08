//
// Created by Mingkai Chen on 2016-12-07.
//

#include "gtest/gtest.h"
#include "graph/tensorless/selector.hpp"
#include "graph/operation/elementary.hpp"


// behavior F300
TEST(SELECTOR, activity_F300)
{
	nnet::varptr<double> A = new nnet::variable<double>(1);
	nnet::varptr<double> B = new nnet::variable<double>(0);
	nnet::placeptr<double> IN = new nnet::placeholder<double>(std::vector<size_t>{1});
	nnet::varptr<double> OP = A - IN; // 1 - x
	nnet::varptr<double> swap = nnet::not_zero(IN, OP);

	// swap = IN
	IN = std::vector<double>{1};
	EXPECT_EQ(IN->get_eval(), swap->get_eval());

	// swap = OP
	IN = std::vector<double>{0};
	EXPECT_EQ(OP->get_eval(), swap->get_eval());

	delete A.get();
	delete B.get();
	delete IN.get();
}


// behavior F301
TEST(SELECTOR, not_zero_F301)
{
}

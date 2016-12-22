//
// Created by Mingkai Chen on 2016-12-07.
//

#include "gtest/gtest.h"
#include "graph/state_selector/conditional.hpp"
#include "graph/state_selector/toggle.hpp"
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
	nnet::placeptr<double> A = 
		new nnet::placeholder<double>(std::vector<size_t>{1});
	nnet::placeptr<double> B = 
		new nnet::placeholder<double>(std::vector<size_t>{1});
	nnet::varptr<double> notZ = nnet::not_zero(A, B);
	
	// Scenario 1: both variables are equal
	// behavior: ideal not_zero should have no preference, 
	// but implementation specifies that first variable A is the default active
	*A = {1.0};
	*B = {1.0};
	EXPECT_EQ(notZ->get_eval(), A->get_eval());
	
	// Scenario 2: B is one, A is zero
	*A = {0.0};
	EXPECT_EQ(notZ->get_eval(), B->get_eval());
	// vice versa
	*A = {1.0};
	*B = {0.0};
	EXPECT_EQ(notZ->get_eval(), A->get_eval());
	
	// Scenario 3: Both zero
	// by implementation, the last one set to zero will NOT be active
	*A = {0.0};
	EXPECT_EQ(notZ->get_eval(), B->get_eval());
	
	delete A.get();
	delete B.get();
}


// behavior F302
TEST(SELECTOR, toggle_active_F302)
{
	std::constant<double>* state_default(1);
	std::placeholder<double> active(std::vector<size_t>{1});
	nnet::toggle<double>* flip = nnet::toggle::build();
}
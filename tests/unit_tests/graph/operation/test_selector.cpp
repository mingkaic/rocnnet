//
// Created by Mingkai Chen on 2016-12-07.
//

#include "gtest/gtest.h"
#include "graph/state_selector/conditional.hpp"
#include "graph/state_selector/push_toggle.hpp"
#include "graph/state_selector/bindable_toggle.hpp"
#include "graph/operation/immutable/elementary.hpp"


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
	// but implementation specifies that first leaf A is the default active
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
TEST(SELECTOR, push_toggle_active_F302)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0, 1);

	double scalar = dis(gen);
	nnet::constant<double>* state_default = nnet::constant<double>::build(scalar);
	nnet::placeholder<double> active(std::vector<size_t>{1});
	nnet::push_toggle<double>* flip = nnet::push_toggle<double>::build(state_default, &active);
	active = { scalar + 1 };
	// by default flip should always be the default
	EXPECT_EQ(state_default->get_eval(), flip->get_eval());

	// inorder to expose active through flip, we need a node with flip as a dependency
	nnet::varptr<double> buffer = nnet::varptr<double>(flip) + 0.0;
	flip->activate(); // active state is propagated to buffer
	EXPECT_EQ(nnet::expose<double>(buffer)[0], scalar + 1);

	// flip's always exposed as default
	EXPECT_EQ(nnet::expose<double>(flip)[0], scalar);
}


// behavior F303
TEST(SELECTOR, push_toggle_deactivate_F303)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0, 1);

	double scalar = dis(gen);
	nnet::constant<double>* state_default = nnet::constant<double>::build(scalar);
	nnet::placeholder<double> active(std::vector<size_t>{1});
	nnet::push_toggle<double>* flip = nnet::push_toggle<double>::build(state_default, &active);
	active = { scalar + 1 };

	// inorder to expose active through flip, we need a node with flip as a dependency
	nnet::varptr<double> buffer = nnet::varptr<double>(flip) + 0.0;
	flip->activate(); // active state is propagated to buffer
	EXPECT_EQ(nnet::expose<double>(buffer)[0], scalar + 1);

	// deactivate
	flip->notify();
	EXPECT_EQ(nnet::expose<double>(buffer)[0], scalar);
}


// behavior F304
TEST(SELECTOR, bindable_toggle_activation_F304)
{
	std::string fake_id = "fake";
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0, 1);

	double scalar = dis(gen);
	double scalar2 = dis(gen);
	nnet::constant<double>* state_default = nnet::constant<double>::build(scalar);
	nnet::constant<double>* state_active = nnet::constant<double>::build(scalar+1);
	nnet::bindable_toggle<double>* flip =
		nnet::bindable_toggle<double>::build(state_default, state_active);
	nnet::varptr<double> buffer = nnet::varptr<double>(flip) + 0.0;

	nnet::constant<double>* state_default2 = nnet::constant<double>::build(scalar2);
	nnet::constant<double>* state_active2 = nnet::constant<double>::build(scalar2+1);
	nnet::bindable_toggle<double>* flip2 =
		nnet::bindable_toggle<double>::build(state_default2, state_active2);
	nnet::varptr<double> buffer2 = nnet::varptr<double>(flip2) + 0.0;

	// bind the toggles
	flip->bind(fake_id, flip2);

	// default state
	EXPECT_EQ(nnet::expose<double>(buffer)[0], scalar);
	EXPECT_EQ(nnet::expose<double>(buffer2)[0], scalar2);

	// active
	flip->activate();
	EXPECT_EQ(nnet::expose<double>(buffer)[0], scalar+1); // buffer is active
	EXPECT_EQ(nnet::expose<double>(buffer2)[0], scalar2);

	// deactivate flip, activate flip2
	flip2->activate();
	EXPECT_EQ(nnet::expose<double>(buffer)[0], scalar); // buffer is deactive
	EXPECT_EQ(nnet::expose<double>(buffer2)[0], scalar2+1); // buffer2 is active

	delete flip;
	delete flip2;
}


// behavior F305
TEST(SELECTOR, bindable_toggle_relationship_F305)
{
	std::string fake_id1 = "fake1";
	std::string fake_id2 = "fake2";
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<double> dis(0, 1);

	std::vector<double> scalars;
	std::vector<nnet::bindable_toggle<double>*> bts;
	std::vector<nnet::varptr<double> > buffers;

	for (size_t i = 0; i < 3; i++)
	{
		double scalar = dis(gen);
		nnet::constant<double>* state_default = nnet::constant<double>::build(scalar);
		nnet::constant<double>* state_active = nnet::constant<double>::build(scalar+1);
		nnet::bindable_toggle<double>* flip =
				nnet::bindable_toggle<double>::build(state_default, state_active);

		scalars.push_back(scalar);
		bts.push_back(flip);
		buffers.push_back(nnet::varptr<double>(flip) + 0.0);
	}

	// relationship 0-1
	bts[1]->bind(fake_id1, bts[0]);
	// relationship 1-2
	bts[1]->bind(fake_id2, bts[2]);

	bts[0]->activate();
	EXPECT_EQ(nnet::expose<double>(buffers[0])[0], scalars[0]+1); // buffer is active
	EXPECT_EQ(nnet::expose<double>(buffers[1])[0], scalars[1]);
	EXPECT_EQ(nnet::expose<double>(buffers[2])[0], scalars[2]);

	// default toggle activate activates the last relationship
	bts[1]->activate();
	EXPECT_EQ(nnet::expose<double>(buffers[0])[0], scalars[0]+1); // buffer 0 is active
	EXPECT_EQ(nnet::expose<double>(buffers[1])[0], scalars[1]+1); // last bound with relationship 1-2, so 0-1's 0 is still active
	EXPECT_EQ(nnet::expose<double>(buffers[2])[0], scalars[2]);

	bts[2]->activate();
	EXPECT_EQ(nnet::expose<double>(buffers[0])[0], scalars[0]+1); // buffer 0 is STILL active
	EXPECT_EQ(nnet::expose<double>(buffers[1])[0], scalars[1]);
	EXPECT_EQ(nnet::expose<double>(buffers[2])[0], scalars[2]+1); // only relationship 1-2, so 0 is still active, but 1 is disabled

	// directed activation
	bts[1]->activate(fake_id1);
	EXPECT_EQ(nnet::expose<double>(buffers[0])[0], scalars[0]); // buffer 0 is now disabled
	EXPECT_EQ(nnet::expose<double>(buffers[1])[0], scalars[1]+1);
	EXPECT_EQ(nnet::expose<double>(buffers[2])[0], scalars[2]+1); // 2 is still active because relationship 1-2 is never touched

	for (nnet::bindable_toggle<double>* btoggles : bts)
	{
		delete btoggles;
	}
}
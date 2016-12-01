//
// Created by Mingkai Chen on 2016-11-17.
//

#include "gtest/gtest.h"
#include "memory/session.hpp"
#include "graph/variable/variable.hpp"
#include "graph/variable/placeholder.hpp"
#include "graph/variable/constant.hpp"


// Session is a singleton. no point in testing constructor, destructor, copy or assign


// Behavior C000
TEST(SESSION, Registration_C000)
{
	nnet::session& sess = nnet::session::get_instance();

	nnet::variable<double> var1(1);
	nnet::placeholder<double> var2(std::vector<size_t>{1});
	nnet::constant<double>* var3 = nnet::constant<double>::build(1);
	
	EXPECT_TRUE(sess.ptr_registered(&var1));
	EXPECT_TRUE(sess.ptr_registered(&var2));
	EXPECT_TRUE(sess.ptr_registered(var3));
	
	delete var3;
}


// Behavior C001
TEST(SESSION, Unregistration_C001)
{
	nnet::session& sess = nnet::session::get_instance();
	nnet::variable<double>* var1 = nullptr;
	nnet::placeholder<double>* var2 = nullptr;
	nnet::constant<double>* var3 = nullptr;
	
	var1 = new nnet::variable<double>(1);
	var2 = new nnet::placeholder<double>(std::vector<size_t>{1});
	var3 = nnet::constant<double>::build(1);
	
	delete var1;
	delete var2;
	delete var3;
	
	EXPECT_FALSE(sess.ptr_registered(var1));
	EXPECT_FALSE(sess.ptr_registered(var2));
	EXPECT_FALSE(sess.ptr_registered(var3));
}


// Behavior C002
TEST(SESSION, Singleton_C002)
{
	nnet::session& sess = nnet::session::get_instance();
	nnet::session& sess1 = nnet::session::get_instance();
	
	ASSERT_EQ(&sess, &sess1);
	// spawn a thread and test addresses there... TODO
}


// Behavior C003
TEST(SESSION, Initialize_C003)
{
	nnet::session& sess = nnet::session::get_instance();
	nnet::const_init<double> dinit(-3.2);
	nnet::const_init<float> finit(3.2);
	nnet::const_init<size_t> cinit(3);
	nnet::variable<double> dvar(std::vector<size_t>{2, 3}, dinit);
	nnet::variable<float> fvar(std::vector<size_t>{2, 3}, finit);
	nnet::variable<size_t> cvar(std::vector<size_t>{2, 3}, cinit);

	EXPECT_FALSE(dvar.is_init());
	EXPECT_FALSE(fvar.is_init());
	EXPECT_FALSE(cvar.is_init());
 	sess.initialize_all<double>();
	EXPECT_TRUE(dvar.is_init());
	EXPECT_FALSE(fvar.is_init());
	EXPECT_FALSE(cvar.is_init());
	sess.initialize_all<float>();
	EXPECT_TRUE(fvar.is_init());
	EXPECT_FALSE(cvar.is_init());
	sess.initialize_all<size_t>();
	EXPECT_TRUE(cvar.is_init());
}
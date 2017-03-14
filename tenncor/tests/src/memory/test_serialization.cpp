//
// Created by Mingkai Chen on 2016-11-17.
//

//#define DISABLE_MEMORY_MODULE_TESTS
#ifndef DISABLE_MEMORY_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"


//#define DISABLE_SERIALIZER_TEST
#ifndef DISABLE_SERIALIZER_TEST


// Behavior C000
TEST(SERIALIZATION, Registration_C000)
{
//	nnet::session& sess = nnet::session::get_instance();
//
//	nnet::variable<double> var1(1);
//	nnet::placeholder<double> var2(std::vector<size_t>{1});
//	nnet::constant<double>* var3 = nnet::constant<double>::build(1);
//
//	EXPECT_TRUE(sess.ptr_registered(&var1));
//	EXPECT_TRUE(sess.ptr_registered(&var2));
//	EXPECT_TRUE(sess.ptr_registered(var3));
//
//	delete var3;
}


// Behavior C001
TEST(SERIALIZATION, Unregistration_C001)
{
//	nnet::session& sess = nnet::session::get_instance();
//	nnet::variable<double>* var1 = nullptr;
//	nnet::placeholder<double>* var2 = nullptr;
//	nnet::constant<double>* var3 = nullptr;
//
//	var1 = new nnet::variable<double>(1);
//	var2 = new nnet::placeholder<double>(std::vector<size_t>{1});
//	var3 = nnet::constant<double>::build(1);
//
//	delete var1;
//	delete var2;
//	delete var3;
//
//	EXPECT_FALSE(sess.ptr_registered(var1));
//	EXPECT_FALSE(sess.ptr_registered(var2));
//	EXPECT_FALSE(sess.ptr_registered(var3));
}


// Behavior C002
TEST(SERIALIZATION, Singleton_C002)
{
//	nnet::session& sess = nnet::session::get_instance();
//	nnet::session& sess1 = nnet::session::get_instance();
//
//	ASSERT_EQ(&sess, &sess1);
//	// spawn a thread and test addresses there... TODO
}


// Behavior C003
TEST(SERIALIZATION, Initialize_C003)
{
//	nnet::session& sess = nnet::session::get_instance();
//	nnet::const_init<double> dinit(-3.2);
//	nnet::const_init<float> finit(3.2);
//	nnet::const_init<size_t> cinit(3);
//	nnet::variable<double> dvar(std::vector<size_t>{2, 3}, dinit);
//	nnet::variable<float> fvar(std::vector<size_t>{2, 3}, finit);
//	nnet::variable<size_t> cvar(std::vector<size_t>{2, 3}, cinit);
//
//	EXPECT_FALSE(dvar.is_init());
//	EXPECT_FALSE(fvar.is_init());
//	EXPECT_FALSE(cvar.is_init());
// 	sess.initialize_all<double>();
//	EXPECT_TRUE(dvar.is_init());
//	EXPECT_FALSE(fvar.is_init());
//	EXPECT_FALSE(cvar.is_init());
//	sess.initialize_all<float>();
//	EXPECT_TRUE(fvar.is_init());
//	EXPECT_FALSE(cvar.is_init());
//	sess.initialize_all<size_t>();
//	EXPECT_TRUE(cvar.is_init());
}


#endif /* DISABLE_SERIALIZER_TEST */


#endif /* DISABLE_MEMORY_MODULE_TESTS */

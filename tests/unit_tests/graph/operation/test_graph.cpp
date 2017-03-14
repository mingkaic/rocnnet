//
// Created by Mingkai Chen on 2016-12-02.
//

#include "gtest/gtest.h"
#include "graph/tensorless/functor.hpp"
#include "graph/operation/immutable/elementary.hpp"


// behavior F200
TEST(FUNCTOR, deletion_var_F200)
{
    nnet::inode<double>* leaf = new nnet::variable<double>(1);
    nnet::inode<double>* leaf2 = new nnet::variable<double>(2);
    nnet::functor<double>* g1 = nnet::functor<double>::build(leaf, 
    [](nnet::varptr<double> leaf)
    {
        return leaf;
    });
    nnet::functor<double>* g2 = g1->append_leaf(leaf2);
    
    std::vector<double> one = nnet::expose<double>(g1);
    std::vector<double> two = nnet::expose<double>(g2);
    
    ASSERT_EQ(1, one.size());
    ASSERT_EQ(1, two.size());
    EXPECT_EQ(1, one[0]);
    EXPECT_EQ(2, two[0]);
    
    // notice the order:
    // if g1 or any leaf is broken by g2 deletion, then delete g1 will fail
    // likewise any leaf deletion will fail if g1 breaks on deletion
    delete g2;
    delete g1;
    delete leaf;
    delete leaf2;
}


TEST(FUNCTOR, deletion_graph_F201)
{
    nnet::inode<double>* leaf = new nnet::variable<double>(1);
    nnet::inode<double>* leaf2 = new nnet::variable<double>(2);
    nnet::functor<double>* g1 = nnet::functor<double>::build(leaf, 
    [](nnet::varptr<double> leaf)
    {
        return leaf;
    });
    nnet::functor<double>* g2 = nnet::functor<double>::build(leaf2, 
    [](nnet::varptr<double> leaf)
    {
        return 1.0-leaf;
    });
    nnet::functor<double>* g3 = g2->append_functor(g1); // should be zero
    
    std::vector<double> zero = nnet::expose<double>(g3);
    
    ASSERT_EQ(1, zero.size());
    EXPECT_EQ(0, zero[0]);
    
    // notice the order:
    // if g1 or any leaf is broken by g2 deletion, then delete g1 will fail
    // likewise any leaf deletion will fail if g1 breaks on deletion
    delete g3;
    delete g2;
    delete g1;
    delete leaf;
    delete leaf2;
}


TEST(FUNCTOR, var_deletion)
{
	nnet::inode<double>* leaf = new nnet::variable<double>(2);
	nnet::functor<double>* g1 = nnet::functor<double>::build(leaf,
	[](nnet::varptr<double> leaf)
	{
		return leaf;
	});

	delete leaf; // this should kill g1
}
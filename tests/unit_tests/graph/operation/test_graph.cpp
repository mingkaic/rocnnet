//
// Created by Mingkai Chen on 2016-12-02.
//

#include "gtest/gtest.h"
#include "graph/tensorless/graph.hpp"
#include "graph/operation/elementary.hpp"


// behavior F200
TEST(GRAPH, deletion_var_F200)
{
    nnet::ivariable<double>* leaf = new nnet::variable<double>(1);
    nnet::ivariable<double>* leaf2 = new nnet::variable<double>(2);
    nnet::graph<double>* g1 = nnet::graph<double>::build(leaf, 
    [](nnet::varptr<double> leaf)
    {
        return leaf;
    });
    nnet::graph<double>* g2 = g1->append_leaf(leaf2);
    
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


TEST(GRAPH, deletion_graph_F201)
{
    nnet::ivariable<double>* leaf = new nnet::variable<double>(1);
    nnet::ivariable<double>* leaf2 = new nnet::variable<double>(2);
    nnet::graph<double>* g1 = nnet::graph<double>::build(leaf, 
    [](nnet::varptr<double> leaf)
    {
        return leaf;
    });
    nnet::graph<double>* g2 = nnet::graph<double>::build(leaf2, 
    [](nnet::varptr<double> leaf)
    {
        return 1.0-leaf;
    });
    nnet::graph<double>* g3 = g2->append_graph(g1); // should be zero
    
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


TEST(GRAPH, var_deletion)
{
	nnet::ivariable<double>* leaf = new nnet::variable<double>(2);
	nnet::graph<double>* g1 = nnet::graph<double>::build(leaf,
	[](nnet::varptr<double> leaf)
	{
		return leaf;
	});

	delete leaf; // this should kill g1
}
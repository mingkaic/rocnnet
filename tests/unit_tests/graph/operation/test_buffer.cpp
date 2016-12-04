//
// Created by Mingkai Chen on 2016-12-02.
//


#include "gtest/gtest.h"
#include "graph/buffer/buffer.hpp"
#include "tensor_test_util.h"
#include "mock_operation.h"
using ::testing::_;

// behavior F200
TEST(BUFFER, DepAccessor_F200) // placeholder name
{
	nnet::placeptr<double> leaf = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "true_leaf");
	nnet::buffer<double>* sep = nnet::buffer<double>::build(leaf, "leaf_buffer");

	leaf = std::vector<double>(6, 1.0);

	nnet::tensorshape lshape = leaf->get_shape();
	nnet::tensorshape sshape = sep->get_shape();
	EXPECT_TRUE(tensorshape_equal(lshape, sshape));
	EXPECT_EQ(leaf->get_eval(), sep->get_eval());
	EXPECT_EQ(leaf->get_gradient(), sep->get_gradient());

	delete leaf.get();
}

// behavior F201
TEST(BUFFER, ReassignDep_F201)
{
	nnet::placeptr<double> leaf = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "leaf");
	nnet::placeptr<double> leaf2 = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "leaf2");

	nnet::buffer<double>* sep = nnet::buffer<double>::build(leaf, "leaf_buffer");
	*sep = *leaf2;
	delete leaf.get();
	// deletion to original leaf will not destroy sep (asserting that deletion behavior works)
	// equivalent to leaf 2 instead of 1
	nnet::tensorshape lshape = leaf2->get_shape();
	nnet::tensorshape sshape = sep->get_shape();
	EXPECT_TRUE(tensorshape_equal(lshape, sshape));
	EXPECT_EQ(leaf2->get_eval(), sep->get_eval());
	EXPECT_EQ(leaf2->get_gradient(), sep->get_gradient());
	delete leaf2.get();
}

// CopyAssign
TEST(BUFFER, CopyAssign)
{
	nnet::placeptr<double> leaf = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "leaf");
	nnet::placeptr<double> leaf2 = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "leaf2");

	nnet::buffer<double>* sep = nnet::buffer<double>::build(leaf, "buffer1");
	nnet::buffer<double>* sep2 = nnet::buffer<double>::build(leaf2, "buffer2");
	*sep = *sep2;
	// both buffers should store the same leaf2 as the dependency
	EXPECT_EQ(leaf2.get(), sep->get());
	EXPECT_EQ(leaf2.get(), sep2->get());

	delete leaf.get();
	delete leaf2.get();
}

// behavior F202
TEST(BUFFER, LeafSetEval_F202)
{
	nnet::placeptr<double> leaf1 = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "leaf");
	nnet::placeptr<double> leaf2 = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "leaf2");
	nnet::placeptr<double> leaf3 = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "leaf2");

	MockOperation* op = MockOperation::build(leaf1, leaf2);
	nnet::buffer<double>* sep = nnet::buffer<double>::build(op, "leaf_buffer");
	sep->leaves_collect([leaf1, leaf2](nnet::ivariable<double>* leaf)
	{
		EXPECT_TRUE(leaf1 == leaf || leaf2 == leaf);
	});
	*sep = *leaf3;
	sep->leaves_collect([leaf3](nnet::ivariable<double>* leaf)
	{
		EXPECT_TRUE(leaf3 == leaf);
	});

	delete leaf1.get();
	delete leaf2.get();
	delete leaf3.get();
}

// behavior F203
TEST(BUFFER, BufferVarSep_F203)
{
	nnet::placeptr<double> leaf = new nnet::placeholder<double>(std::vector<size_t>{1, 2, 3}, "true_leaf");
	nnet::buffer<double>* sep = nnet::buffer<double>::build(leaf, "leaf_buffer");

	MockOperation* op = MockOperation::build(leaf, sep);
	EXPECT_CALL(*op, mock_update(_, _)).Times(2);
	leaf = std::vector<double>(6, 1.0);

	delete leaf.get();
}
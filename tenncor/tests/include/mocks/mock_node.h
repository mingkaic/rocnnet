//
// Created by Mingkai Chen on 2017-03-16.
//

#ifndef TENNCOR_MOCK_NODE_H
#define TENNCOR_MOCK_NODE_H

#include <algorithm>

#include "util_test.h"
#include "gmock/gmock.h"

#include "graph/inode.hpp"

using namespace nnet;


class mock_node : public inode<double>
{
public:
	mock_node (std::string name = "") : inode<double>(name) {}
	mock_node (const mock_node& other) : inode<double>(other) {}
	mock_node (mock_node&& other) : inode<double>(std::move(other)) {}
	mock_node& operator = (const mock_node& other)
	{
		inode<double>::operator = (other);
		return *this;
	}
	mock_node& operator = (mock_node&& other)
	{
		inode<double>::operator = (std::move(other));
		return *this;
	}

	MOCK_CONST_METHOD0(get_shape, tensorshape(void));
	MOCK_CONST_METHOD0(good_status, bool(void));
	MOCK_CONST_METHOD0(get_eval, const tensor<double>*(void));
	MOCK_CONST_METHOD1(get_leaves, void(GRAD_CACHE&));
	MOCK_METHOD1(get_gradient, const tensor<double>*(inode<double>*));
	MOCK_METHOD1(get_leaf, inode<double>*(variable<double>*));

protected:
	virtual inode<double>* clone_impl (void) const { return new mock_node(*this); }
	virtual inode<double>* move_impl (void) { return new mock_node(std::move(*this)); }
};


#endif //TENNCOR_MOCK_NODE_H

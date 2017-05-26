//
// Created by Mingkai Chen on 2017-04-19.
//

#ifndef TENNCOR_MOCK_LEAF_H
#define TENNCOR_MOCK_LEAF_H

#include "gtest/gtest.h"
#include "util_test.h"
#include "fuzz.h"

#include "graph/leaf/ileaf.hpp"

using namespace nnet;


class mock_leaf : public ileaf<double>
{
public:
	mock_leaf (std::string name) : ileaf<double>(random_def_shape(), name) {}
	mock_leaf (const tensorshape& shape, std::string name) : ileaf<double>(shape, name) {}

	virtual inode<double>* get_gradient (inode<double>*) { return nullptr; }
	virtual inode<double>* get_leaf (variable<double>*) { return nullptr; }
	virtual void get_leaves (typename inode<double>::GRAD_CACHE&) const {}

	void set_good (void) { this->is_init_ = true; }
	void mock_init_data (initializer<double>& initer) { initer(this->data_); }

protected:
	virtual inode<double>* clone_impl (void) const { return new mock_leaf(*this); }
	virtual inode<double>* move_impl (void) { return new mock_leaf(std::move(*this)); }
};


#endif //TENNCOR_MOCK_LEAF_H

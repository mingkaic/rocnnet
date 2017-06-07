//
// Created by Mingkai Chen on 2017-04-19.
//

#ifndef TENNCOR_MOCK_IVARIABLE_H
#define TENNCOR_MOCK_IVARIABLE_H

#include "util_test.h"
#include "fuzz.h"

#include "graph/leaf/ivariable.hpp"

using namespace nnet;


class mock_ivariable : public ivariable<double>
{
public:
	mock_ivariable (const tensorshape& shape,
		initializer<double>* init,
		std::string name) : ivariable<double>(shape, init, name) {}

	virtual void get_leaf (inode<double>*&, variable<double>*) {}
	virtual void get_leaves (typename inode<double>::GRAD_CACHE&) const {}

	initializer<double>* get_initializer (void) { return this->init_; }

protected:
	virtual inode<double>* clone_impl (void) const { return new mock_ivariable(*this); }
	virtual inode<double>* move_impl (void) { return new mock_ivariable(std::move(*this)); }
};

#endif //TENNCOR_MOCK_IVARIABLE_H

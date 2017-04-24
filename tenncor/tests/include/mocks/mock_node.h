//
// Created by Mingkai Chen on 2017-03-16.
//

#ifndef TENNCOR_MOCK_NODE_H
#define TENNCOR_MOCK_NODE_H

#include <algorithm>

#include "util_test.h"

#include "graph/inode.hpp"

using namespace nnet;


// not a substitute for leaves...
// WARNING: deletes data
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

	~mock_node (void)
	{
		if (data_) delete data_;
	}

	tensor<double>* data_ = nullptr;

	virtual tensorshape get_shape (void) const { return data_->get_shape(); }
	virtual bool good_status (void) const { return nullptr != data_; }
	virtual const tensor<double>* get_eval(void) const { return data_; }
	virtual void get_leaves (GRAD_CACHE&) const {}
	virtual inode<double>* get_gradient (inode<double>* wrt) { return wrt == this ? this : nullptr; }
	virtual inode<double>* get_leaf (variable<double>*) { return nullptr; }

protected:
	virtual inode<double>* clone_impl (void) const { return new mock_node(*this); }
	virtual inode<double>* move_impl (void) { return new mock_node(std::move(*this)); }
};


#endif //TENNCOR_MOCK_NODE_H

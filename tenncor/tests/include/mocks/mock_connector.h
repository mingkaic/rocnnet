//
// Created by Mingkai Chen on 2017-03-17.
//

#ifndef TENNCOR_MOCK_CONNECTOR_H
#define TENNCOR_MOCK_CONNECTOR_H

#include <algorithm>

#include "util_test.h"
#include "mockerino.h"

#include "graph/connector/iconnector.hpp"

using namespace nnet;


class mock_connector : public iconnector<double>, public mocker
{
public:
	mock_connector (std::vector<inode<double>*> dependencies, std::string label) :
		iconnector<double>(dependencies, label) {}

	mock_connector (const mock_connector& other) :
		iconnector<double>(other) {}

	mock_connector (mock_connector&& other) :
		iconnector<double>(std::move(other)) {}

	mock_connector& operator = (const mock_connector& other)
	{
		iconnector<double>::operator = (other);
		return *this;
	}

	mock_connector& operator = (mock_connector&& other)
	{
		iconnector<double>::operator = (std::move(other));
		return *this;
	}

	void* get_gid (void) { return this->gid_; }

	virtual void temporary_eval (const iconnector<double>*,inode<double>*&) const {}
	virtual tensorshape get_shape (void) const { return tensorshape(); }
	virtual bool good_status (void) const { return false; }
	virtual const tensor<double>* get_eval (void) const { return nullptr; }
	virtual void get_leaf (inode<double>*&, variable<double>*) {}
	virtual varptr<double> get_gradient (inode<double>*) { return nullptr; }
	virtual bool read_proto (const tenncor::tensor_proto&) { return false; }

	virtual void update (subject*)
	{
		label_incr("update1");
	}

	virtual void commit_sudoku (void)
	{
		label_incr("commit_sudoku");
	}
	virtual void get_leaves (GRAD_CACHE&) const
	{
		label_incr("get_leaves1");
	}

protected:
	virtual inode<double>* clone_impl (void) const
	{
		return new mock_connector(*this);
	}
	virtual inode<double>* move_impl (void)
	{
		return new mock_connector(std::move(*this));
	}

	virtual std::vector<typename iconnector<double>::conn_summary> summarize (void) const { return {}; }
};


#endif //TENNCOR_MOCK_CONNECTOR_H

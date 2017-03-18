//
// Created by Mingkai Chen on 2017-03-17.
//

#ifndef TENNCOR_MOCK_CONNECTOR_H
#define TENNCOR_MOCK_CONNECTOR_H

#include <algorithm>

#include "util_test.h"
#include "gmock/gmock.h"

#include "graph/connector/iconnector.hpp"

using namespace nnet;


class mock_connector : public iconnector<double>
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

	MOCK_CONST_METHOD2(temporary_eval,
		void(const iconnector<double>*,tensor<double>*&));
	MOCK_CONST_METHOD0(get_shape, tensorshape(void));
	MOCK_CONST_METHOD0(good_status, bool(void));
	MOCK_CONST_METHOD0(get_eval, const tensor<double>*(void));
	MOCK_CONST_METHOD1(get_gradient, const tensor<double>*(inode<double>*));
	MOCK_CONST_METHOD1(get_leaves, void(GRAD_CACHE&));
	MOCK_METHOD1(get_leaf, inode<double>*(variable<double>*));
	MOCK_METHOD1(update, void(subject*));
	MOCK_METHOD0(commit_sudoku, void(void));

protected:
	virtual inode<double>* clone_impl (void) const
	{
		return new mock_connector(*this);
	}
	virtual inode<double>* move_impl (void)
	{
		return new mock_connector(std::move(*this));
	}
};


#endif //TENNCOR_MOCK_CONNECTOR_H

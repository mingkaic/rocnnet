//
// Created by Mingkai Chen on 2017-03-15.
//

#ifndef TENNCOR_MOCK_SUBJECT_H
#define TENNCOR_MOCK_SUBJECT_H

#include <algorithm>

#include "util_test.h"
#include "gmock/gmock.h"

#include "graph/react/subject.hpp"

using namespace nnet;


class mock_subject : public subject
{
public:
	mock_subject (void) {}
	mock_subject (iobserver* obs) { this->attach(obs, 0); }
	mock_subject (iobserver* obs, iobserver* obs2)
	{
		this->attach(obs, 0);
		this->attach(obs2, 1);
	}
	~mock_subject (void) {}

	mock_subject (const mock_subject& other) : subject(other) {}
	mock_subject (mock_subject&& other) : subject(std::move(other)) {}

	mock_subject& operator = (const mock_subject& other)
	{
		subject::operator = (other);
		return *this;
	}
	mock_subject& operator = (mock_subject&& other)
	{
		subject::operator = (std::move(other));
		return *this;
	}

	void mock_attach(iobserver* viewer, size_t idx)
	{
		subject::attach(viewer, idx);
	}

	void mock_detach(iobserver* obs, signed i = -1)
	{
		if (i < 0)
		{
			subject::detach(obs);
		}
		else
		{
			subject::detach(obs, (size_t)i);
		}
	}

	MOCK_METHOD1(detach, void(iobserver*));
	MOCK_METHOD2(detach, void(iobserver*,size_t));
};


class mock_subject2 : public subject {};


#endif //TENNCOR_MOCK_SUBJECT_H

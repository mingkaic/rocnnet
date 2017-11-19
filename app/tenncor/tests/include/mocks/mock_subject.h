//
// Created by Mingkai Chen on 2017-03-15.
//

#ifndef TENNCOR_MOCK_SUBJECT_H
#define TENNCOR_MOCK_SUBJECT_H

#include <algorithm>

#include "util_test.h"
#include "mockerino.h"

#include "graph/react/subject.hpp"

using namespace nnet;


class mock_subject : public subject, public mocker
{
public:
	mock_subject (void) {}

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

	virtual void detach (iobserver*)
	{
		label_incr("detach1");
	}

	virtual void detach (iobserver*,size_t)
	{
		label_incr("detach2");
	}

	virtual std::string get_label (void) const
	{
		return "mock_title";
	}
};


// mock sub to expose subject constructor (which is protected by default)
class mock_subject2 : public subject
{
public:
	virtual std::string get_label (void) const
	{
		return "mock_title";
	}
};


#endif //TENNCOR_MOCK_SUBJECT_H

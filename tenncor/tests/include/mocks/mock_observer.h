//
// Created by Mingkai Chen on 2017-03-15.
//

#ifndef TENNCOR_MOCK_OBSERVER_H
#define TENNCOR_MOCK_OBSERVER_H

#include <algorithm>

#include "util_test.h"
#include "mockerino.h"

#include "graph/react/iobserver.hpp"

using namespace nnet;

class dummy_observer : public iobserver
{
public:
	dummy_observer (void) : iobserver() {}
	dummy_observer (subject* arg) : iobserver({arg})  {}
	dummy_observer (subject* a, subject* b) : iobserver({a, b})  {}
	~dummy_observer (void) {}

	dummy_observer (dummy_observer&& other) : iobserver(std::move(other)) {}

	// dummy stops us from executing these real actions,
	// since we test using dummy attach/detach procedures
	virtual void update (subject*) {}
	virtual void update (size_t, notification) {}
	virtual void commit_sudoku (void) {}

protected:
	dummy_observer (const dummy_observer& other) : iobserver(other) {}
};

class mock_observer : public dummy_observer, public mocker
{
public:
	// trust tester to allocate on stack
	mock_observer (void) : dummy_observer() {}
	mock_observer (subject* arg) : dummy_observer({arg})  {}
	mock_observer (subject* a, subject* b) : dummy_observer({a, b})  {}
	~mock_observer (void) {}

	std::vector<subject*> expose_dependencies (void)
	{
		return this->dependencies_;
	}

	virtual void update (subject*)
	{
		label_incr("update1");
	}

	virtual void update (size_t,notification)
	{
		label_incr("update2");
	}

	virtual void commit_sudoku (void)
	{
		label_incr("commit_sudoku");
	}
};


class mock_observer2 : public iobserver, public mocker
{
public:
	// trust tester to allocate on stack
	mock_observer2 (void) : iobserver() {}
	mock_observer2 (subject* arg) : iobserver({arg})  {}
	mock_observer2 (subject* a, subject* b) : iobserver({a, b})  {}
	~mock_observer2 (void) {}

	mock_observer2 (const mock_observer2& other) : iobserver(other) {}
	mock_observer2 (mock_observer2&& other) : iobserver(std::move(other)) {}
	mock_observer2& operator = (const mock_observer2& other)
	{
		iobserver::operator = (other);
		return *this;
	}
	mock_observer2& operator = (mock_observer2&& other)
	{
		iobserver::operator = (std::move(other));
		return *this;
	}

	void mock_add_dependency (subject* dep)
	{
		this->add_dependency(dep);
	}

	void mock_remove_dependency (size_t idx)
	{
		this->remove_dependency(idx);
	}

	void mock_replace_dependency (subject* dep, size_t idx)
	{
		this->replace_dependency(dep, idx);
	}

	std::vector<subject*> expose_dependencies (void)
	{
		return this->dependencies_;
	}

	virtual void update (subject*)
	{
		label_incr("update1");
	}

	virtual void commit_sudoku (void)
	{
		label_incr("commit_sudoku");
	}
};


#endif //TENNCOR_MOCK_OBSERVER_H

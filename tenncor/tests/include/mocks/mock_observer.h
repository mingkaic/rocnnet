//
// Created by Mingkai Chen on 2017-03-15.
//

#ifndef TENNCOR_MOCK_OBSERVER_H
#define TENNCOR_MOCK_OBSERVER_H

#include <algorithm>

#include "util_test.h"
#include "gmock/gmock.h"

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

	virtual void update (subject*) {}
	virtual void update (size_t,notification) {}
	virtual void commit_sudoku (void) {}

protected:
	dummy_observer (const dummy_observer& other) : iobserver(other) {}
};

class mock_observer : public dummy_observer
{
public:
	// trust tester to allocate on stack
	mock_observer (void) : dummy_observer() {}
	mock_observer (subject* arg) : dummy_observer({arg})  {}
	mock_observer (subject* a, subject* b) : dummy_observer({a, b})  {}
	~mock_observer (void) {}

	mock_observer* clone (void) const { return new mock_observer(*this); }
	mock_observer (mock_observer&& other) : dummy_observer(std::move(other)) {}

	std::vector<subject*> expose_dependencies (void)
	{
		return this->dependencies_;
	}
	MOCK_METHOD1(update, void(subject*));
	MOCK_METHOD2(update, void(size_t,notification));
	MOCK_METHOD0(commit_sudoku, void(void));

protected:
	mock_observer (const mock_observer& other) : dummy_observer(other) {}
};


#endif //TENNCOR_MOCK_OBSERVER_H

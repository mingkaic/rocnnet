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

class dummy_observer : public iobserver, public mocker
{
public:
	dummy_observer (void) : iobserver() {}
	dummy_observer (subject* arg) : iobserver(std::vector<subject*>{arg})  {}
	dummy_observer (subject* a, subject* b) : iobserver({a, b})  {}
	dummy_observer (std::vector<subject*> args) : iobserver(args)  {}
	~dummy_observer (void) {}

	dummy_observer (dummy_observer&& other) : iobserver(std::move(other)) {}
	
	void mock_add_dependency (subject* dep)
	{
		this->add_dependency(dep);
	}

	void mock_clear_dependency (void)
	{
		this->dependencies_.clear();
	}

	std::vector<subject*> expose_dependencies (void)
	{
		return this->dependencies_;
	}

	virtual void update (std::unordered_set<size_t>)
	{
		label_incr("update1");
	}
	
	virtual void death_on_broken (void)
	{
		label_incr("death_on_broken");
	}

protected:
	dummy_observer (const dummy_observer& other) : iobserver(other) {}
};

class mock_observer : public dummy_observer
{
public:
	// trust tester to allocate on stack
	mock_observer (void) : dummy_observer() {}
	mock_observer (subject* arg) : dummy_observer(arg)  {}
	mock_observer (subject* a, subject* b) : dummy_observer(a, b)  {}
	mock_observer (std::vector<subject*> args) : dummy_observer(args)  {}
	~mock_observer (void) {}

	using dummy_observer::update;
	virtual void update (std::unordered_set<size_t> callers, notification msg)
	{
		if (msg == notification::UNSUBSCRIBE)
		{
			for (size_t callidx : callers)
			{
				this->remove_dependency(callidx);
			}
		}
		label_incr("update2");
	}
};


class mock_observer2 : public dummy_observer
{
public:
	// trust tester to allocate on stack
	mock_observer2 (void) : dummy_observer() {}
	mock_observer2 (subject* arg) : dummy_observer(arg)  {}
	mock_observer2 (subject* a, subject* b) : dummy_observer(a, b)  {}
	~mock_observer2 (void) {}

	mock_observer2 (const mock_observer2& other) : dummy_observer(other) {}
	mock_observer2 (mock_observer2&& other) : dummy_observer(std::move(other)) {}
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

	void mock_remove_dependency (size_t idx)
	{
		this->remove_dependency(idx);
	}

	void mock_replace_dependency (subject* dep, size_t idx)
	{
		this->replace_dependency(dep, idx);
	}
};


#endif //TENNCOR_MOCK_OBSERVER_H

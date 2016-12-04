//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ccoms/iobserver.hpp"

#ifdef subject_hpp

namespace ccoms
{

// REACTIVE NODE

bool reactive_node::safe_destroy (void)
{
	if (suicidal())
	{
		// deletion logic, change here if we allow stack allocation in the future
		delete this;
		return true;
	}
	return false;
}

reactive_node::~reactive_node (void)
{
	for (void** ptrr : ptrrs_)
	{
		*ptrr = nullptr;
	}
}

void reactive_node::set_death (void** ptr)
{
	ptrrs_.emplace(ptr);
}
void reactive_node::unset_death (void** ptr)
{
	ptrrs_.erase(ptr);
}

// SUBJECT

void subject::attach (iobserver* viewer, size_t idx)
{
	audience_[viewer].push_back(idx);
}

void subject::detach (iobserver* viewer)
{
	audience_.erase(viewer);
	if (suicidal() && audience_.empty())
	{
		safe_destroy();
	}
}

subject::subject (subject_owner* owner)
{
	var_ = owner;
}

subject::~subject (void)
{
	auto it = audience_.begin();
	while (audience_.end() != it)
	{
		// when an observer is destroyed, 
		// the observer attempts to detach itself from its subjects
		// that's why we increment iterator before we delete
		iobserver* captive = it->first;
		it++;
		captive->safe_destroy(); // flag captive for destruction
	}
}

subject::subject (const subject& other, subject_owner* owner)
{
	var_ = owner;
}

void subject::notify (update_message msg)
{
	caller_info info(this);
	// everyone get the same message
	for (auto it : audience_)
	{
		for (size_t idx : it.second)
		{
			info.caller_idx_ = idx;
			iobserver *viewer = it.first;
			viewer->update(info, msg);
		}
	}
}

bool subject::no_audience (void) const
{
	return audience_.empty();
}

bool subject::safe_destroy (void)
{
	if (var_)
	{
		delete var_;
		return true;
	}
	return reactive_node::safe_destroy();
}

subject_owner* subject::get_owner (void) {
	return var_;
}

// SUBJECT OWNER

void subject_owner::copy (const subject_owner& other)
{
	caller_ = new subject(*other.caller_, this);
}

subject_owner::subject_owner (const subject_owner& other)
{
	copy(other);
}

subject_owner::subject_owner (void) { caller_ = new subject(this); }

subject_owner::~subject_owner (void) { delete caller_; }

void subject_owner::notify (subject_owner* grad)
{
	ccoms::update_message msg;
	if (grad)
	{
		msg.grad_ = grad->caller_;
	}
	caller_->notify(msg);
}

void subject_owner::notify (ccoms::update_message msg)
{
	caller_->notify(msg);
}

bool subject_owner::no_audience (void) const
{
	return caller_->no_audience();
}

}

#endif
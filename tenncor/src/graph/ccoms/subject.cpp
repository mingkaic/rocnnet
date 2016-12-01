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

void subject::notify (subject* grad)
{
	update_message msg(this);
	msg.grad_ = grad;
	// everyone get the same message
	for (auto it : audience_)
	{
		for (size_t idx : it.second)
		{
			msg.caller_idx_ = idx;
			iobserver *viewer = it.first;
			viewer->update(msg);
		}
	}
}

bool subject::no_audience (void) const
{
	return audience_.empty();
}

}

#endif
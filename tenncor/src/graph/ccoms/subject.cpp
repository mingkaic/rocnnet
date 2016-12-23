//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ccoms/iobserver.hpp"
#include <iostream>

#ifdef subject_hpp

namespace ccoms
{

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
	if (nullptr != var_)
	{
		// safe_destroy is meant as a means of suicide
		// killing owner will in turn kill this
		delete var_;
		return true;
	}
	return ireactive_node::safe_destroy();
}

subject_owner*& subject::get_owner (void) {
	return var_;
}

}

#endif
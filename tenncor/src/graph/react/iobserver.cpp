//
//  iobserver.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/react/iobserver.hpp"

#ifdef TENNCOR_IOBSERVER_HPP

namespace nnet
{

iobserver::~iobserver (void)
{
	size_t ndeps = dependencies_.size();
	for (size_t i = 0; i < ndeps; i++)
	{
		remove_dependency(i);
	}
}

iobserver& iobserver::operator = (const iobserver& other)
{
	if (this != &other)
	{
		copy_helper(other);
	}
	return *this;
}

iobserver& iobserver::operator = (iobserver&& other)
{
	if (this != &other)
	{
		move_helper(std::move(other));
	}
	return *this;
}

iobserver::iobserver (void) {}

iobserver::iobserver (std::vector<subject*> dependencies)
{
	for (subject* dep : dependencies)
	{
		add_dependency(dep);
	}
}

iobserver::iobserver (const iobserver& other)
{
	copy_helper(other);
}

iobserver::iobserver (iobserver&& other)
{
	move_helper(std::move(other));
}

void iobserver::add_dependency (subject* dep)
{
	if (dep)
	{
		dep->attach(this, dependencies_.size());
	}
	dependencies_.push_back(dep);
}

void iobserver::remove_dependency (size_t idx)
{
	if (idx > dependencies_.size()) return;
	if (subject* sub = dependencies_[idx])
	{
		sub->detach(this);
		// update dependencies
		dependencies_[idx] = nullptr;
		while (false == dependencies_.empty() &&
			nullptr == dependencies_.back())
		{
			dependencies_.pop_back();
		}
	}
}

void iobserver::replace_dependency (subject* dep, size_t idx)
{
	size_t ndeps = dependencies_.size();
	if (dep)
	{
		dep->attach(this, idx);
	}
	if (idx >= ndeps)
	{
		dependencies_.insert(dependencies_.end(), idx - ndeps + 1, nullptr);
	}
	else if (subject* lastsub = dependencies_[idx])
	{
		lastsub->detach(this, idx);
	}
	dependencies_[idx] = dep;
}

void iobserver::update (size_t dep_idx, notification msg)
{
	switch (msg)
	{
		case UNSUBSCRIBE:
			remove_dependency(dep_idx);
			commit_sudoku();
			break;

		case UPDATE:
			update(dependencies_[dep_idx]); // value update
			break;
	}
}

void iobserver::copy_helper (const iobserver& other)
{
	for (subject* sub : dependencies_)
	{
		sub->detach(this);
	}
	dependencies_.clear();
	for (subject* dep : other.dependencies_)
	{
		add_dependency(dep);
	}
}

void iobserver::move_helper (iobserver&& other)
{
	for (subject* sub : dependencies_)
	{
		sub->detach(this);
	}
	dependencies_ = std::move(other.dependencies_);
	// replace subs audience from other to this
	for (size_t i = 0, n = dependencies_.size();
		i < n; i++)
	{
		dependencies_[i]->detach(&other, i);
		dependencies_[i]->attach(this, i);
	}
	size_t ndeps = dependencies_.size();
	for (size_t i = 0; i < ndeps; i++)
	{
		dependencies_[i]->attach(this, i);
	}
}

}

#endif

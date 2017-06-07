//
//  iobserver.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/react/iobserver.hpp"

#ifdef EDGE_RCD
#include "edgeinfo/comm_record.hpp"
#endif /* EDGE_RCD */

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

#ifdef EDGE_RCD

// record subject-object edge
if (rocnnet_record::erec::rec_good) rocnnet_record::erec::rec.node_release(this);

#endif /* EDGE_RCD */
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

bool iobserver::has_subject (subject* sub) const
{
	auto et = dependencies_.end();
	return et != std::find(dependencies_.begin(), et, sub);
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
	size_t depsize = dependencies_.size();
	if (idx >= depsize)
	{
		throw std::logic_error(nnutils::formatter() << "attempting to remove argument index "
			<< idx << " from observer with " << depsize << " arguments");
	}
	if (subject* sub = dependencies_[idx])
	{
		sub->detach(this, idx);
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
		if (lastsub != dep) lastsub->detach(this, idx);
	}
	dependencies_[idx] = dep;
}

void iobserver::update (std::unordered_set<size_t> dep_indices, notification msg)
{
	for (size_t dep_idx : dep_indices)
	{
		switch (msg)
		{
			case UNSUBSCRIBE:
				remove_dependency(dep_idx);
				break;
			case UPDATE:
				update(dependencies_[dep_idx]); // value update
				break;
		}
	}
	if (UNSUBSCRIBE == msg)
	{
		death_on_broken();
	}
}

void iobserver::copy_helper (const iobserver& other)
{
	for (size_t i = 0, n = dependencies_.size(); i < n; i++)
	{
		if (dependencies_[i]) dependencies_[i]->detach(this, i);
	}
	dependencies_.clear();
	for (subject* dep : other.dependencies_)
	{
		if (dep) add_dependency(dep);
	}
}

void iobserver::move_helper (iobserver&& other)
{
	for (size_t i = 0, n = dependencies_.size(); i < n; i++)
	{
		if (dependencies_[i]) dependencies_[i]->detach(this, i);
	}
	dependencies_ = std::move(other.dependencies_);
	// replace subs audience from other to this
	for (size_t i = 0, n = dependencies_.size();
		i < n; i++)
	{
		// attach before detaching to ensure dep i doesn't suicide
		if (dependencies_[i])
		{
			dependencies_[i]->attach(this, i);
			dependencies_[i]->detach(&other, i);
		}
	}
}

}

#endif

//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/react/iobserver.hpp"

#ifdef TENNCOR_SUBJECT_HPP

#ifdef EDGE_RCD

#include "edgeinfo/grpc_vis_record.hpp"

#endif /* EDGE_RCD */

namespace nnet
{

subject::~subject (void)
{
	std::unordered_set<iobserver*> killers = killers_;
	for (iobserver* killer : killers)
	{
		killer->remove_ondeath_dependent(this);
	}

	notify(UNSUBSCRIBE); // unsubscribe all audiences

#ifdef EDGE_RCD

// record subject-object edge
if (rocnnet_record::record_status::rec_good)
{
	rocnnet_record::record_status::rec->node_release(this);
}

#endif /* EDGE_RCD */
}

subject& subject::operator = (const subject&) { return *this; }

subject& subject::operator = (subject&& other)
{
	if (this != &other)
	{
		std::unordered_map<iobserver*, std::unordered_set<size_t> > temp = other.audience_;
		for (auto& audpair : temp)
		{
			iobserver* aud = audpair.first;
			for (size_t idx : audpair.second)
			{
				aud->replace_dependency(this, idx);
			}
		}
		other.audience_.clear();
	}
	return *this;
}

void subject::notify (notification msg) const
{
#ifdef RPC_RCD
if (rocnnet_record::record_status::rec_good && UPDATE == msg)
{
	if (rocnnet_record::rpc_record* grpc =
		dynamic_cast<rocnnet_record::rpc_record*>(
		rocnnet_record::record_status::rec.get()))
	{
		grpc->data_update(this);
	}
}
#endif /* RPC_RCD */
	std::vector<iobserver*> obs;
	for (auto it = audience_.begin(), et = audience_.end(); it != et; it++)
	{
		obs.push_back(it->first);
	}

	for (iobserver* viewer : obs)
	{
		auto it = audience_.find(viewer);
		if (it != audience_.end())
		{
			viewer->update(it->second, msg);
		}
	}
}

bool subject::no_audience (void) const
{
	return audience_.empty();
}

size_t subject::n_audience (void) const { return audience_.size(); }

void subject::steal_observers (subject* other)
{
	std::unordered_map<iobserver*, std::unordered_set<size_t> > aud_cpy = other->audience_;
	for (auto aud_pair : aud_cpy)
	{
		iobserver* aud = aud_pair.first;
		for (size_t i : aud_pair.second)
		{
			aud->replace_dependency(this, i);
		}
	}
}

subject::subject (void) {}

void subject::death_on_noparent (void) {}

void subject::attach_killer (iobserver* killer)
{
	if (killer && killers_.end() == killers_.find(killer))
	{
		killers_.insert(killer);
		killer->add_ondeath_dependent(this);
	}
}

void subject::detach_killer (iobserver* killer)
{
	if (killer && killers_.end() != killers_.find(killer))
	{
		killers_.erase(killer);
		killer->remove_ondeath_dependent(this);
	}
}

subject::subject (const subject&) {}

subject::subject (subject&& other)
{
	std::unordered_map<iobserver*, std::unordered_set<size_t> > temp = other.audience_;
	for (auto& audpair : temp)
	{
		iobserver* aud = audpair.first;
		for (size_t idx : audpair.second)
		{
			aud->replace_dependency(this, idx);
		}
	}
	other.audience_.clear();
}

void subject::attach (iobserver* viewer, size_t idx)
{
#ifdef EDGE_RCD

// record subject-object edge
if (rocnnet_record::record_status::rec_good)
{
	rocnnet_record::record_status::rec->edge_capture(viewer, this, idx);
}

#endif /* EDGE_RCD */

	audience_[viewer].emplace(idx);
}

void subject::detach (iobserver* viewer)
{
#ifdef EDGE_RCD

if (rocnnet_record::record_status::rec_good)
{
	// record subject-object edge
	for (size_t idx : audience_[viewer])
	{
		rocnnet_record::record_status::rec->edge_release(viewer, this, idx);
	}
}

#endif /* EDGE_RCD */

	audience_.erase(viewer);
	if (audience_.empty())
	{
		death_on_noparent();
	}
}

void subject::detach (iobserver* viewer, size_t idx)
{
#ifdef EDGE_RCD

// record subject-object edge
if (rocnnet_record::record_status::rec_good)
{
	rocnnet_record::record_status::rec->edge_release(viewer, this, idx);
}

#endif /* EDGE_RCD */

	auto it = audience_.find(viewer);
	if (audience_.end() != it)
	{
		it->second.erase(idx);
		if (it->second.empty())
		{
			audience_.erase(viewer);
		}
	}
	if (audience_.empty())
	{
		death_on_noparent();
	}
}

}

#endif
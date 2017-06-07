//
// Created by Mingkai Chen on 2017-01-30.
//

#include "edgeinfo/comm_record.hpp"

#ifdef comm_record_hpp

namespace rocnnet_record
{

#ifdef EDGE_RCD
edge_record erec::rec(EDGE_RCD);
bool erec::rec_good = true;
#endif

edge_record::~edge_record (void)
{
	erec::rec_good = false;
}

void edge_record::edge_capture (nnet::iobserver* obs, nnet::subject* sub, size_t idx)
{
	std::vector<nnet::subject*>& subs = edges_[obs];
	if (subs.size() <= idx)
	{
		subs.insert(subs.end(), idx-subs.size()+1, nullptr);
	}
	subs[idx] = sub;
	subset_[sub].emplace(obs);
}

void edge_record::edge_release (nnet::iobserver* obs, nnet::subject* sub, size_t idx)
{
	auto it = edges_.find(obs);
	if (edges_.end() != it)
	{
		std::vector<nnet::subject*>& subs = it->second;
		if (subs.size() <= idx)
		{
			throw std::exception();
		}
		if (sub == subs[idx]) subs[idx] = nullptr;
	}
}

void edge_record::node_release (nnet::subject* sub)
{
	subset_.find(sub);
	std::unordered_set<nnet::iobserver*>& auds = subset_[sub];
	for (nnet::iobserver* ob : auds)
	{
		auto it = edges_.find(ob);
		if (edges_.end() != it)
		{
			std::vector<nnet::subject*>& subs = it->second;
			for (nnet::subject*& s : subs)
			{
				if (s == sub) s = nullptr;
			}
		}
	}
	subset_.erase(sub);
}

void edge_record::node_release (nnet::iobserver* obs)
{
	edges_.erase(obs);
}

}

#endif
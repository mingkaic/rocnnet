//
// Created by mingkaichen on 2017-11-08.
//

#include "edgeinfo/adjlist_record.hpp"

#ifdef ADJLIST_RECORD_HPP

namespace rocnnet_record
{

adjlist_record::~adjlist_record (void) {}

void adjlist_record::node_release (const nnet::subject* sub)
{
	subj_nodes.erase(sub);
}

void adjlist_record::edge_capture (const nnet::iobserver* obs, const nnet::subject* sub, size_t idx)
{
	if (!obs->is_recordable())
	{
		return;
	}
	auto it = subj_nodes.find(sub);
	if (subj_nodes.end() == it)
	{
		auto insert_info = subj_nodes.insert({ sub, std::unordered_set<obs_info, obs_info_hash>() });
		it = insert_info.first;
	}
	it->second.emplace(obs_info{obs, idx});
}

void adjlist_record::edge_release (const nnet::iobserver* obs, const nnet::subject* sub, size_t idx)
{
	auto it = subj_nodes.find(sub);
	if (subj_nodes.end() != it)
	{
		it->second.erase(obs_info{obs, idx});
	}
}

bool operator == (const typename adjlist_record::obs_info& lhs, const typename adjlist_record::obs_info& rhs)
{
	return (lhs.obs_ == rhs.obs_) && (lhs.idx_ == rhs.idx_);
}

}

#endif

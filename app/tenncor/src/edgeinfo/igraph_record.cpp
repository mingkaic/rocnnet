//
// Created by Ming Kai Chen on 2017-11-08.
//

#include "edgeinfo/igraph_record.hpp"

#ifdef igraph_record_hpp

namespace rocnnet_record
{

igraph_record::~igraph_record (void)
{
	record_status::rec_good = false;
}

void igraph_record::node_capture (nnet::subject* sub)
{
	subj_nodes.insert({ sub, std::unordered_set<obs_info, obs_info_hash>() });
}

void igraph_record::node_release (nnet::subject* sub)
{
	subj_nodes.erase(sub);
}

void igraph_record::edge_capture (nnet::iobserver* obs, nnet::subject* sub, size_t idx)
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

void igraph_record::edge_release (nnet::iobserver* obs, nnet::subject* sub, size_t idx)
{
	auto it = subj_nodes.find(sub);
	if (subj_nodes.end() != it)
	{
		it->second.erase(obs_info{obs, idx});
	}
}

bool operator == (const typename igraph_record::obs_info& lhs, const typename igraph_record::obs_info& rhs)
{
	return (lhs.obs_ == rhs.obs_) && (lhs.idx_ == rhs.idx_);
}

}

#endif

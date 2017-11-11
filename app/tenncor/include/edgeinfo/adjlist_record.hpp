//
//  grpc_vis_record.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-07.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "edgeinfo/igraph_record.hpp"

#pragma once
#ifdef EDGE_RCD

#ifndef adjlist_record_hpp
#define adjlist_record_hpp

namespace rocnnet_record
{

class adjlist_record : public igraph_record
{
public:
	struct obs_info
	{
		const nnet::iobserver* obs_;
		size_t idx_;

		obs_info (const nnet::iobserver* obs, size_t idx) :
			obs_(obs), idx_(idx) {}
	};

	struct obs_info_hash {
		size_t operator () (const obs_info& info) const
		{
			std::hash<std::string> strhash;
			return strhash(nnutils::formatter() << info.idx_ << info.obs_);
		}
	};

	virtual ~adjlist_record (void);

	// all nodes are subjects
	virtual void node_release (const nnet::subject* sub);

	virtual void edge_capture (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx);

	virtual void edge_release (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx);

protected:
	std::unordered_map<const nnet::subject*,std::unordered_set<obs_info, obs_info_hash> > subj_nodes;
};

bool operator == (const adjlist_record::obs_info& lhs, const adjlist_record::obs_info& rhs);

}

#endif /* adjlist_record_hpp */

#endif /* EDGE_RCD */

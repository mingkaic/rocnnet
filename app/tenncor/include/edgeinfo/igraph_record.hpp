//
//  igraph_record.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-07.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "graph/inode.hpp"
#include "utils/utils.hpp"

#pragma once
#if defined(CSV_RCD) || defined(VIS_RCD)

#ifndef igraph_record_hpp
#define igraph_record_hpp

namespace rocnnet_record
{

class igraph_record
{
public:
	struct obs_info
	{
		nnet::iobserver* obs_;
		size_t idx_;

		obs_info (nnet::iobserver* obs, size_t idx) :
				obs_(obs), idx_(idx) {}
	};

	struct obs_info_hash {
		size_t operator () (const obs_info& info) const
		{
			std::hash<std::string> strhash;
			return strhash(nnutils::formatter() << info.idx_ << info.obs_);
		}
	};

	virtual ~igraph_record (void);

	void node_capture (nnet::subject* sub); // all nodes are subjects

	void node_release (nnet::subject* sub);

	void edge_capture (nnet::iobserver* obs, nnet::subject* sub, size_t obs_idx);

	void edge_release (nnet::iobserver* obs, nnet::subject* sub, size_t obs_idx);

protected:
	std::unordered_map<nnet::subject*, std::unordered_set<obs_info, obs_info_hash> > subj_nodes;
};

bool operator == (const igraph_record::obs_info& lhs, const igraph_record::obs_info& rhs);

struct record_status
{
	static std::unique_ptr<igraph_record> rec;
	static bool rec_good;
};

}

#endif /* igraph_record_hpp */

#endif /* CSV_RCD || VIS_RCD */

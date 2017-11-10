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
#ifdef EDGE_RCD

#ifndef igraph_record_hpp
#define igraph_record_hpp

namespace rocnnet_record
{

class igraph_record
{
public:
	virtual ~igraph_record (void);

	virtual void node_capture (const nnet::subject* sub) = 0; // all nodes are subjects

	virtual void node_release (const nnet::subject* sub) = 0;

	virtual void edge_capture (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx) = 0;

	virtual void edge_release (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx) = 0;
};

struct record_status
{
	static std::unique_ptr<igraph_record> rec;
	static bool rec_good;
};

}

#endif /* igraph_record_hpp */

#endif /* EDGE_RCD */

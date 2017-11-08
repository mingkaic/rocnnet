//
//  igraph_record.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-07.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "graph/inode.hpp"

#pragma once
#ifdef EDGE_RCD

#ifndef igraph_record_hpp
#define igraph_record_hpp

namespace rocnnet_record
{

class igraph_record
{
public:
	virtual void node_capture (nnet::subject* sub) = 0; // all nodes are subjects

	virtual void node_release (nnet::subject* sub) = 0;

	virtual void edge_capture (nnet::iobserver* obs, nnet::subject* sub, size_t obs_idx) = 0;

	virtual void edge_release (nnet::iobserver* obs, nnet::subject* sub, size_t obs_idx) = 0;

};

}

#endif /* igraph_record_hpp */

#endif /* EDGE_RCD */

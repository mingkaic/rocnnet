//
//  grpc_vis_record.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-07.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#pragma once
#ifdef EDGE_RCD

#include <thread>

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>
#include "proto/grpc_gui.grpc.pb.h"

#include "edgeinfo/igraph_record.hpp"
#include "edgeinfo/grpc_vis/gui_notifier.hpp"

#ifndef GRPC_VIS_RECORD_HPP
#define GRPC_VIS_RECORD_HPP

namespace rocnnet_record
{

class rpc_record final : public igraph_record
{
public:
	rpc_record (std::string host, size_t port);

	virtual void node_release (const nnet::subject* sub);

	void data_update (const nnet::subject* sub);

	virtual void edge_capture (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx);

	virtual void edge_release (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx);

private:
	nnet::shareable_cache<std::string, visor::NodeMessage> node_cache_;

	nnet::shareable_cache<std::string, visor::EdgeMessage> edge_cache_;

	std::thread server_;
};

}

#endif /* GRPC_VIS_RECORD_HPP */

#endif /* EDGE_RCD */

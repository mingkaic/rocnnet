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

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>
#include "proto/grpc_cli.grpc.pb.h"

#ifndef grpc_vis_record_hpp
#define grpc_vis_record_hpp

namespace rocnnet_record
{

using common_res = grpc::ClientAsyncResponseReader<visor::Empty>;
using rpc_call = std::function<std::unique_ptr<common_res>(
	grpc::ClientContext*, grpc::CompletionQueue*)>;

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
	void send_message (rpc_call call);

	std::unordered_set<std::string> notifiable_;

	std::unique_ptr<visor::GUINotifier::Stub> stub_;
};

}

#endif /* grpc_vis_record_hpp */

#endif /* EDGE_RCD */

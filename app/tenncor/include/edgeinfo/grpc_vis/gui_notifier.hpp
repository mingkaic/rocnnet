//
//  gui_notifier.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-12.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#pragma once
#ifdef EDGE_RCD

#include <experimental/optional>

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>
#include "proto/grpc_gui.grpc.pb.h"

#include "thread/shareable_cache.hpp"

using namespace std::experimental;

#pragma once
#ifndef gui_notifier_hpp
#define gui_notifier_hpp

namespace rocnnet_record
{

class gui_notifier final : public visor::GUINotifier::Service
{
public:
	gui_notifier (
		nnet::shareable_cache<std::string,visor::NodeMessage>& node_src,
		nnet::shareable_cache<std::string,visor::EdgeMessage>& edge_src);

	grpc::Status SubscribeNode (grpc::ServerContext* context,
		const visor::ClientId* client,
		grpc::ServerWriter<visor::NodeMessage>* writer) override;

	grpc::Status SubscribeEdge (grpc::ServerContext* context,
		const visor::ClientId* client,
		grpc::ServerWriter<visor::EdgeMessage>* writer) override;

	grpc::Status EndSubscription (grpc::ServerContext* context,
		const visor::ClientId* client, visor::Empty* out) override;

private:
	// shared with graph_vis_record
	nnet::shareable_cache<std::string,visor::NodeMessage>* node_src_;
	nnet::shareable_cache<std::string,visor::EdgeMessage>* edge_src_;

	// own
	std::unordered_set<std::string> subscriptions_;
};

void spawn_server (std::string addr,
	nnet::shareable_cache<std::string,visor::NodeMessage>& node_src,
	nnet::shareable_cache<std::string,visor::EdgeMessage>& edge_src);

}

#endif /* gui_notifier_hpp */

#endif /* EDGE_RCD */

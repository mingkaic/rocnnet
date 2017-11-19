//
// Created by Ming Kai Chen on 2017-11-11.
//

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

#include "proto/grpc_gui.grpc.pb.h"

#include "utils/utils.hpp"

#pragma once
#ifndef gui_consumer_hpp
#define gui_consumer_hpp

namespace tenncor_graph
{

class gui_consumer final
{
public:
	gui_consumer (std::shared_ptr<grpc::Channel> channel);

	void SubscribeNode (void);

	void SubscribeEdge (void);

	void EndSubscription (void);

private:
	const std::string id_ = nnutils::uuid(this);

	std::unique_ptr<visor::GUINotifier::Stub> stub_;
};

}

#endif /* gui_consumer_hpp */

//
// Created by Ming Kai Chen on 2017-11-11.
//

#include <sstream>

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

#include "proto/grpc_cli.grpc.pb.h"

#pragma once
#ifndef graph_gui_hpp
#define graph_gui_hpp

namespace tenncor_graph
{

class graph_gui final
{
public:
	~graph_gui (void);

	void run (std::string host, size_t port);

private:
	class CallHandler;

	class NodeHandler;

	class EdgeHandler;

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	std::unique_ptr<grpc::Server> server_;

	visor::GUINotifier::AsyncService service_;
};

}

#endif /* graph_gui_hpp */

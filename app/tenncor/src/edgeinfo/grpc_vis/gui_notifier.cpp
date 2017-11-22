//
//  Created by Ming Kai Chen on 2017-11-12.
//

#include <thread>

#include "thread/stop_flag.hpp"
#include "edgeinfo/grpc_vis/gui_notifier.hpp"

#ifdef gui_notifier_hpp

namespace rocnnet_record
{

gui_notifier::gui_notifier (nodecache_t& node_src, edgecache_t& edge_src) :
	node_src_(&node_src), edge_src_(&edge_src) {}

grpc::Status gui_notifier::SubscribeNode (grpc::ServerContext*,
	const visor::StringId* client,
	grpc::ServerWriter<visor::NodeMessage>* writer)
{
	std::string cli_id = client->id();
	subscriptions_.emplace(cli_id);
	auto et = subscriptions_.end();
	std::shared_ptr<nnet::cache_node<visor::NodeMessage> > iter = nullptr;

	while (et != subscriptions_.find(cli_id))
	{
		if (optional<visor::NodeMessage> msg = node_src_->get_latest(iter))
		{
			writer->Write(*msg);
		}
	}
	return grpc::Status::OK;
}

grpc::Status gui_notifier::SubscribeEdge (grpc::ServerContext*,
	const visor::StringId* client,
	grpc::ServerWriter<visor::EdgeMessage>* writer)
{
	std::string cli_id = client->id();
	subscriptions_.emplace(cli_id);
	auto et = subscriptions_.end();
	std::shared_ptr<nnet::cache_node<visor::EdgeMessage> > iter = nullptr;

	while (et != subscriptions_.find(cli_id))
	{
		if (optional<visor::EdgeMessage> msg = edge_src_->get_latest(iter))
		{
			writer->Write(*msg);
		}
	}
	return grpc::Status::OK;
}

grpc::Status gui_notifier::EndSubscription (grpc::ServerContext*,
	const visor::StringId* client, visor::Empty*)
{
	subscriptions_.erase(client->id());
	return grpc::Status::OK;
}


grpc::Status gui_notifier::GetNodeData (grpc::ServerContext* context,
	const visor::StringId* node, visor::NodeData* out)
{
	std::string node_id = node->id();
	return grpc::Status::OK;
}

void gui_notifier::clear_subscriptions (void)
{
	subscriptions_.clear();
	node_src_->escape_wait();
	edge_src_->escape_wait();
}

void spawn_server (std::string addr, nodecache_t& node_src, edgecache_t& edge_src)
{
	gui_notifier service(node_src, edge_src);

	grpc::ServerBuilder builder;
	builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
	builder.RegisterService(&service);

	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
	std::cout << "server listening on " << addr << std::endl;
	std::thread server_wait([&server]
	{
		server->Wait();
	});

	nnet::stop_wait(); // wait until stop flag is set
	service.clear_subscriptions();
	server->Shutdown();
	server_wait.join();
}

}

#endif

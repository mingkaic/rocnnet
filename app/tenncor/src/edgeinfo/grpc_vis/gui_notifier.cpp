//
//  Created by Ming Kai Chen on 2017-11-12.
//

#include "edgeinfo/grpc_vis/gui_notifier.hpp"

#ifdef gui_notifier_hpp

namespace rocnnet_record
{

gui_notifier::gui_notifier (
	nnet::shareable_cache<std::string, visor::NodeMessage>& node_src,
	nnet::shareable_cache<std::string, visor::EdgeMessage>& edge_src) :
node_src_(&node_src), edge_src_(&edge_src) {}

grpc::Status gui_notifier::SubscribeNode (grpc::ServerContext*,
	const visor::ClientId* client,
	grpc::ServerWriter<visor::NodeMessage>* writer)
{
	std::string cli_id = client->id();
	subscriptions_.emplace(cli_id);
	auto et = subscriptions_.end();
	std::shared_ptr<nnet::cache_node<visor::NodeMessage> > iter = nullptr;

	while (et != subscriptions_.find(cli_id))
	{
		visor::NodeMessage msg = node_src_->get_latest(iter);
		writer->Write(msg);
	}
	return grpc::Status::OK;
}

grpc::Status gui_notifier::SubscribeEdge (grpc::ServerContext*,
	const visor::ClientId* client,
	grpc::ServerWriter<visor::EdgeMessage>* writer)
{
	std::string cli_id = client->id();
	subscriptions_.emplace(cli_id);
	auto et = subscriptions_.end();
	std::shared_ptr<nnet::cache_node<visor::EdgeMessage> > iter = nullptr;

	while (et != subscriptions_.find(cli_id))
	{
		visor::EdgeMessage msg = edge_src_->get_latest(iter);
		writer->Write(msg);
	}
	return grpc::Status::OK;
}

grpc::Status gui_notifier::EndSubscription (grpc::ServerContext*,
	const visor::ClientId* client, visor::Empty*)
{
	subscriptions_.erase(client->id());
	return grpc::Status::OK;
}

void spawn_server (std::string addr,
	nnet::shareable_cache<std::string,visor::NodeMessage>& node_src,
	nnet::shareable_cache<std::string,visor::EdgeMessage>& edge_src)
{
	gui_notifier service(node_src, edge_src);

	grpc::ServerBuilder builder;
	builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
	builder.RegisterService(&service);

	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
	std::cout << "server listening on " << addr << std::endl;
	server->Wait();
}

}

#endif

//
// Created by Ming Kai Chen on 2017-11-11.
//

#include <thread>
#include <sstream>

#include "gui_consumer.hpp"

#ifdef gui_consumer_hpp

namespace tenncor_graph
{

gui_consumer::gui_consumer (std::shared_ptr<grpc::Channel> channel) :
	stub_(visor::GUINotifier::NewStub(channel)) {}

void gui_consumer::SubscribeNode (void)
{
	visor::ClientId client;
	visor::NodeMessage res;
	grpc::ClientContext context;

	client.set_id(id_);
	std::unique_ptr<grpc::ClientReader<visor::NodeMessage> > reader(
			stub_->SubscribeNode(&context, client));

	while (reader->Read(&res))
	{
		std::string id = res.id();
		visor::NodeMessage_NodeStatus status = res.status();
		if (visor::NodeMessage_NodeStatus::NodeMessage_NodeStatus_REMOVE == status)
		{
			std::cout << id << " Removed" << std::endl;
		}
		else // if (visor::NodeMessage_NodeStatus::NodeMessage_NodeStatus_UPDATE == status)
		{
			std::cout << id << " Updated" << std::endl;
		}
	}
	grpc::Status status = reader->Finish();
	if (status.ok())
	{
		std::cout << "Node subscription terminated OK" << std::endl;
	}
	else
	{
		std::cout << "Node subscription termination failed" << std::endl;
	}
}

void gui_consumer::SubscribeEdge (void)
{
	visor::ClientId client;
	visor::EdgeMessage res;
	grpc::ClientContext context;

	client.set_id(id_);
	std::unique_ptr<grpc::ClientReader<visor::EdgeMessage> > reader(
			stub_->SubscribeEdge(&context, client));

	while (reader->Read(&res))
	{
		std::string oid = res.obsid();
		std::string sid = res.subid();
		size_t idx = res.idx();
		visor::EdgeMessage_EdgeStatus status = res.status();
		if (visor::EdgeMessage_EdgeStatus::EdgeMessage_EdgeStatus_ADD == status)
		{
			std::cout << oid  << ", " << sid << " @ " << idx << " Added" << std::endl;
		}
		else // if (visor::EdgeMessage_EdgeStatus::EdgeMessage_EdgeStatus_REMOVE == status)
		{
			std::cout << oid  << ", " << sid << " @ " << idx << " Removed" << std::endl;
		}
	}
	grpc::Status status = reader->Finish();
	if (status.ok())
	{
		std::cout << "Edge subscription terminated OK" << std::endl;
	}
	else
	{
		std::cout << "Edge subscription termination failed" << std::endl;
	}
}

void gui_consumer::EndSubscription (void)
{
	visor::ClientId client;
	visor::Empty res;
	grpc::ClientContext context;

	client.set_id(id_);
	grpc::Status status = stub_->EndSubscription(&context, client, &res);
	if (status.ok())
	{
		std::cout << "Terminated Success" << std::endl;
	}
	else
	{
		std::cout << "Termination Failed" << std::endl;
	}
}

}

int main(int argc, char** argv)
{
	using namespace std::chrono_literals;

	std::stringstream addr;
	addr << "localhost:" << 50981;
	tenncor_graph::gui_consumer consumer(grpc::CreateChannel(addr.str(), grpc::InsecureChannelCredentials()));

	std::thread nodeHandler([](tenncor_graph::gui_consumer& consumer)
	{
		consumer.SubscribeNode();
	}, std::ref(consumer));

	std::thread edgeHandler([](tenncor_graph::gui_consumer& consumer)
	{
		consumer.SubscribeEdge();
	}, std::ref(consumer));

//	std::this_thread::sleep_for(30s);
//	consumer.EndSubscription();

	nodeHandler.join();
	edgeHandler.join();

	return 0;
}

#endif

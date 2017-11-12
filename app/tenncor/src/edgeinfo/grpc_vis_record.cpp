//
// Created by Ming Kai Chen on 2017-11-07.
//

#include "edgeinfo/grpc_vis_record.hpp"

#ifdef grpc_vis_record_hpp

namespace rocnnet_record
{

#ifdef RPC_RCD

std::unique_ptr<igraph_record> record_status::rec =
	std::make_unique<rpc_record>("localhost", 10980);
bool record_status::rec_good = true;

#endif /* RPC_RCD */

rpc_record::rpc_record (std::string host, size_t port)
{
	stub_ = visor::GUINotifier::NewStub(grpc::CreateChannel(
		nnutils::formatter() << host << ":" << port,
		grpc::InsecureChannelCredentials()));
}

void rpc_record::node_release (const nnet::subject* sub)
{
	visor::NodeMessage req;
	req.set_id(sub->get_uid());
	req.set_status(visor::NodeMessage_NodeStatus::NodeMessage_NodeStatus_REMOVE);
	send_message(
	[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
	{
		return this->stub_->PrepareAsyncNodeChange(ctx, req, cq);
	});
}

void rpc_record::data_update (const nnet::subject* sub)
{
	std::string sid = sub->get_uid();
	if (notifiable_.end() == notifiable_.find(sid))
	{
		return;
	}
	visor::NodeMessage req;
	req.set_id(sid);
	req.set_status(visor::NodeMessage_NodeStatus::NodeMessage_NodeStatus_UPDATE);
	send_message(
	[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
	{
		return this->stub_->PrepareAsyncNodeChange(ctx, req, cq);
	});
}

void rpc_record::edge_capture (const nnet::iobserver* obs,
	const nnet::subject* sub, size_t obs_idx)
{
	if (!obs->is_recordable())
	{
		return;
	}
	visor::EdgeMessage req;
	req.set_obsid(obs->get_uid());
	req.set_subid(sub->get_uid());
	req.set_idx(obs_idx);
	req.set_status(visor::EdgeMessage_EdgeStatus::EdgeMessage_EdgeStatus_ADD);
	send_message(
	[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
	{
		return this->stub_->PrepareAsyncEdgeChange(ctx, req, cq);
	});
}

void rpc_record::edge_release (const nnet::iobserver* obs,
						   const nnet::subject* sub, size_t obs_idx)
{
	if (!obs->is_recordable())
	{
		return;
	}
	visor::EdgeMessage req;
	req.set_obsid(obs->get_uid());
	req.set_subid(sub->get_uid());
	req.set_idx(obs_idx);
	req.set_status(visor::EdgeMessage_EdgeStatus::EdgeMessage_EdgeStatus_REMOVE);
	send_message(
	[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
	{
		return this->stub_->PrepareAsyncEdgeChange(ctx, req, cq);
	});
}

void rpc_record::add_notifiable (const nnet::subject* sub)
{
	notifiable_.emplace(sub);
}

void rpc_record::send_message (rpc_call call)
{
	grpc::ClientContext context;
	grpc::CompletionQueue cq;
	grpc::Status status;
	visor::Empty reply;

	std::unique_ptr<common_res> rpc(call(&context, &cq));
	rpc->StartCall();
	rpc->Finish(&reply, &status, (void*)1);
}

}

#endif

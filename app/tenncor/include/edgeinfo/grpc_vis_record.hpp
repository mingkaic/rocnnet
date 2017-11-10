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
#include "proto/grpc_vis.pb.h"

#ifndef grpc_vis_record_hpp
#define grpc_vis_record_hpp

namespace rocnnet_record
{

using common_res = grpc::ClientAsyncResponseReader<Empty>;

class rpc_record final : public igraph_record
{
public:
	rpc_record (std::string host, size_t port)
	{
		grpc::Channel channel = grpc::CreateChannel(
			nnutils::formatter() << host << ":" << port,
			grpc::InsecureCredentials());
		stub_ = visor::GUINotifier::NewStub(channel);
	}

	// return true if message is ok
	template <typename REQ>
	bool send_message (
		std::function<void(REQ&)> set_req,
		std::function<common_res*(grpc::ClientContext*,
			REQ, grpc::CompletionQueue*)> call)
	{
		grpc::ClientContext context;
		grpc::CompletionQueue cq;
		grpc::Status status;

		visor::Empty reply;
		REQ request;
		set_req(request);

		std::unique_ptr<common_res> rpc = call(&context, request, &cq);
		rpc->StartCall();
		rpc->Finish(&reply, &status, (void*)1);

		void* got_tag;
		bool ok = false;
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*)1);
		GPR_ASSERT(ok);

		return status.ok();
	}

	virtual void node_capture (const nnet::subject* sub)
	{
		bool ok = send_message(
		[sub](visor::NodeAdd& req)
		{
			req->set_nodeId(sub->get_uid());
			req->set_nodeLabel(sub->get_label());
		},
		[this](grpc::ClientContext* ctx, REQ req, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncAddNode(ctx, req, cq);
		});
	}

	virtual void node_release (const nnet::subject* sub)
	{
		bool ok = send_message(
		[sub](visor::NodeRemove& req)
		{
			req->set_nodeId(sub->get_uid());
		},
		[this](grpc::ClientContext* ctx, REQ req, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncRmNode(ctx, req, cq);
		});
	}

	void data_update (const nnet::subject* sub)
	{
		std::vector<double> data = { 0 };
		if (nnet::inode<double>* dnode = dynamic_cast<nnet::inode<double>*>(sub))
		{
			data = nnet::expose(dnode);
		}
		bool ok = send_message(
		[sub, &data](visor::NodeUpdate& req)
		{
			req->set_nodeId(sub->get_uid());
			req->set_data(data);
		},
		[this](grpc::ClientContext* ctx, REQ req, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncUpdateNode(ctx, req, cq);
		});
	}

	virtual void edge_capture (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx)
	{
		bool ok = send_message(
		[obs, sub, obs_idx](visor::EdgeMessage& req)
		{
			req->set_obsId(obs->get_uid());
			req->set_subId(sub->get_uid());
			req->set_idx(obs_idx);
		},
		[this](grpc::ClientContext* ctx, REQ req, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncAddEdge(ctx, req, cq);
		});
	}

	virtual void edge_release (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx)
	{
		bool ok = send_message(
		[obs, sub, obs_idx](visor::EdgeMessage& req)
		{
			req->set_obsId(obs->get_uid());
			req->set_subId(sub->get_uid());
			req->set_idx(obs_idx);
		},
		[this](grpc::ClientContext* ctx, REQ req, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncRmEdge(ctx, req, cq);
		});
	}

private:
	std::unique_ptr<visor::GUINotifier::Stub> stub_;
};

}

#endif /* grpc_vis_record_hpp */

#endif /* EDGE_RCD */

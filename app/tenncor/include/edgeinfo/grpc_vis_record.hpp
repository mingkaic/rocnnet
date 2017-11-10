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
#include "proto/grpc_vis.grpc.pb.h"

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
	rpc_record (std::string host, size_t port)
	{
		stub_ = visor::GUINotifier::NewStub(grpc::CreateChannel(
			nnutils::formatter() << host << ":" << port,
			grpc::InsecureChannelCredentials()));
	}

	// return true if message is ok
	bool send_message (rpc_call call)
	{
		grpc::ClientContext context;
		grpc::CompletionQueue cq;
		grpc::Status status;

		visor::Empty reply;

		std::unique_ptr<common_res> rpc(call(&context, &cq));
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
		visor::NodeAdd req;
		req.set_nodeid(sub->get_uid());
		req.set_nodelabel(sub->get_label());
		bool ok = send_message(
		[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncAddNode(ctx, req, cq);
		});
	}

	virtual void node_release (const nnet::subject* sub)
	{
		visor::NodeRemove req;
		req.set_nodeid(sub->get_uid());
		bool ok = send_message(
		[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncRmNode(ctx, req, cq);
		});
	}

	void data_update (const nnet::subject* sub)
	{
		std::vector<double> data = { 0 };
		if (nnet::inode<double>* dnode = const_cast<nnet::inode<double>*>(
			dynamic_cast<const nnet::inode<double>*>(sub)))
		{
			data = nnet::expose(dnode);
		}
		visor::NodeUpdate req;
		req.set_nodeid(sub->get_uid());
		for (double d : data)
		{
			req.add_data(d);
		}
		bool ok = send_message(
		[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncUpdateNode(ctx, req, cq);
		});
	}

	virtual void edge_capture (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx)
	{
		visor::EdgeMessage req;
		const nnet::subject* ofroms = dynamic_cast<const nnet::subject*>(obs);
		assert(ofroms);
		req.set_obsid(ofroms->get_uid());
		req.set_subid(sub->get_uid());
		req.set_idx(obs_idx);
		bool ok = send_message(
		[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
		{
			return this->stub_->PrepareAsyncAddEdge(ctx, req, cq);
		});
	}

	virtual void edge_release (const nnet::iobserver* obs,
		const nnet::subject* sub, size_t obs_idx)
	{
		visor::EdgeMessage req;
		const nnet::subject* ofroms = dynamic_cast<const nnet::subject*>(obs);
		assert(ofroms);
		req.set_obsid(ofroms->get_uid());
		req.set_subid(sub->get_uid());
		req.set_idx(obs_idx);
		bool ok = send_message(
		[this, &req](grpc::ClientContext* ctx, grpc::CompletionQueue* cq)
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

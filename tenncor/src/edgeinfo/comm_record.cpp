//
// Created by Mingkai Chen on 2017-01-30.
//

#include "edgeinfo/comm_record.hpp"

#ifdef comm_record_hpp

namespace rocnnet_record
{

#ifdef EDGE_RCD
edge_record erec::rec(EDGE_RCD);
#endif

ptr_record edge_record::prec_;

bool operator == (edge_record::subinfo const& lhs, edge_record::subinfo const& rhs)
{
	return (lhs.obs_ == rhs.obs_) && (lhs.sub_ == rhs.sub_) && (lhs.sid_ == rhs.sid_);
}

void edge_record::edge_capture (nnet::iobserver* obs, nnet::subject* sub, size_t idx)
{
	edge_record::prec_.add(obs);
	edge_record::prec_.add(sub);
	edges_.insert(subinfo(obs, sub, idx));
}

void edge_record::edge_release (nnet::iobserver* obs, nnet::subject* sub, size_t idx)
{
	auto it = edges_.find(subinfo(obs, sub, idx));
	if (edges_.end() != it)
	{
		edge_record::prec_.remove(obs);
		edge_record::prec_.remove(sub);
	}
}

void edge_record::node_release (nnet::subject* sub)
{
	// linear search and remove all edges from sub
	auto it = edges_.begin(), et = edges_.end();
	while (it != et)
	{
		bool releasenode = (*it).sub_ == sub;
		auto cpyit = it;
		it++;
		if (releasenode)
		{
			edges_.erase(cpyit);
		}
	}
}

void edge_record::node_release (nnet::iobserver* obs)
{
	// linear search and remove all edges to obs
	auto it = edges_.begin(), et = edges_.end();
	while (it != et)
	{
		bool releasenode = (*it).obs_ == obs;
		auto cpyit = it;
		it++;
		if (releasenode)
		{
			edges_.erase(cpyit);
		}
	}
}

}

#endif
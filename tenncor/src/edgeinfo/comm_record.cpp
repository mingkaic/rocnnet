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

void edge_record::edge_capture (
		iobserver* observer,
		subject_owner* subject,
		size_t idx)
{
	edge_record::prec_.add(observer);
	edge_record::prec_.add(subject);
	edges_.insert(subinfo(observer, subject, idx));
}

void edge_record::edge_release (
		iobserver* observer,
		subject_owner* subject,
		size_t idx)
{
	auto it = edges_.find(subinfo(observer, subject, idx));
	if (edges_.end() != it)
	{
		edge_record::prec_.remove(observer);
		edge_record::prec_.remove(subject);
	}
}

}

#endif
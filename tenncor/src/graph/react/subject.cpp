//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/react/iobserver.hpp"

#ifdef EDGE_RCD
#include "edgeinfo/comm_record.hpp"
#endif /* EDGE_RCD */

#ifdef TENNCOR_SUBJECT_HPP

namespace nnet
{

subject::~subject (void)
{
	notify(UNSUBSCRIBE); // unsubscribe all audiences
}

subject& subject::operator = (const subject&) { return *this; }

subject& subject::operator = (subject&& other)
{
	if (this != &other)
	{
		audience_ = std::move(other.audience_);
	}
	return *this;
}

void subject::notify (notification msg) const
{
	auto it = audience_.begin();
	while (audience_.end() != it)
	{
		iobserver* viewer = it->first;
		std::unordered_set<size_t> indices = it->second;
		// increment iterator before updating to account for audience_ modifications
		it++;
		viewer->update(indices, msg);
	}
}

bool subject::no_audience (void) const
{
	return audience_.empty();
}

subject::subject (const subject&) {}

subject::subject (subject&& other) :
	audience_(std::move(other.audience_)) {}

void subject::attach (iobserver* viewer, size_t idx)
{

#ifdef EDGE_RCD
// record subject-object edge
rocnnet_record::erec::rec.edge_capture(viewer, this, idx);
#endif /* EDGE_RCD */

	audience_[viewer].emplace(idx);
}

void subject::detach (iobserver* viewer)
{

#ifdef EDGE_RCD
// record subject-object edge
for (size_t idx : audience_[viewer])
{
	rocnnet_record::erec::rec.edge_release(viewer, this, idx);
}
#endif /* EDGE_RCD */

	audience_.erase(viewer);
	if (audience_.empty())
	{
		commit_sudoku_sub();
	}
}

void subject::detach (iobserver* viewer, size_t idx)
{

#ifdef EDGE_RCD
// record subject-object edge
rocnnet_record::erec::rec.edge_release(viewer, this, idx);
#endif /* EDGE_RCD */

	auto it = audience_.find(viewer);
	if (audience_.end() != it)
	{
		it->second.erase(idx);
	}
	if (it->second.empty())
	{
		audience_.erase(viewer);
	}
	if (audience_.empty())
	{
		commit_sudoku_sub();
	}
}

}

#endif
//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ccoms/iobserver.hpp"

#ifdef EDGE_RCD
#include "edgeinfo/comm_record.hpp"
#endif /* EDGE_RCD */

#ifdef subject_hpp

namespace ccoms
{

void subject::attach (iobserver* viewer, size_t idx)
{
#ifdef EDGE_RCD
// record subject-object edge
rocnnet_record::erec::rec.edge_capture(viewer, var_, idx);
#endif /* EDGE_RCD */
	audience_[viewer].push_back(idx);
}

void subject::detach (iobserver* viewer)
{
#ifdef EDGE_RCD
// record subject-object edge
for (size_t idx : audience_[viewer])
{
	rocnnet_record::erec::rec.edge_release(viewer, var_, idx);
}
#endif /* EDGE_RCD */
	audience_.erase(viewer);
	if (suicidal() && audience_.empty())
	{
		safe_destroy();
	}
}

subject::subject (subject_owner* owner)
{
	var_ = owner;
}

subject::~subject (void)
{
	auto it = audience_.begin();
	while (audience_.end() != it)
	{
		// when an observer is destroyed, 
		// the observer attempts to detach itself from its subjects
		// that's why we increment iterator before we delete
		iobserver* captive = it->first;
		it++;
		captive->safe_destroy(); // flag captive for destruction
	}
}

subject::subject (const subject& other, subject_owner* owner)
{
	var_ = owner;
}

void subject::notify (update_message msg)
{
	caller_info info(this);
	// everyone get the same message
	auto it = audience_.begin();
	while (audience_.end() != it)
	{
		// we're asking our audience to unsubscribe.
		// we only need to ask once
		if (msg.cmd_ == update_message::REMOVE_ARG)
		{
			iobserver* aud = it->first;
			// subscribing detaches it from audience_, so we increment before
			it++;
			aud->update(info, msg);
		}
		else
		{
			for (size_t idx : it->second) {
				info.caller_idx_ = idx;
				iobserver *viewer = it->first;
				viewer->update(info, msg);
			}
			it++;
		}
	}
}

bool subject::no_audience (void) const
{
	return audience_.empty();
}

bool subject::safe_destroy (void)
{
	if (nullptr != var_)
	{
		// safe_destroy is meant as a means of suicide
		// killing owner will in turn kill this
		delete var_;
		return true;
	}
	return ireactive_node::safe_destroy();
}

subject_owner*& subject::get_owner (void) {
	return var_;
}

}

#endif
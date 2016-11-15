//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ccoms/subject.hpp"
#include "graph/ccoms/iobserver.hpp"

#ifdef subject_hpp

namespace ccoms
{
	
void subject::merge_leaves (std::unordered_set<ccoms::subject*>& src)
{
	src.emplace(this);
}

bool subject::no_audience (void)
{
	return audience_.empty();
}

subject::~subject (void)
{
	auto it = audience_.begin();
	while (audience_.end() != it)
	{
		iobserver* captive = *it;
		it++;
		delete captive;
	}
}

void subject::attach (iobserver* viewer)
{
	audience_.emplace(viewer);
}

void subject::detach (iobserver* viewer)
{
	audience_.erase(viewer);
}

void subject::notify (subject* caller)
{
	for (iobserver* viewer : audience_)
	{
		viewer->update(caller);
	}
}

}

#endif
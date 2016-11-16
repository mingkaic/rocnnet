//
//  iobserver.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ccoms/iobserver.hpp"

#ifdef observer_hpp

namespace ccoms
{
		
void iobserver::add_dependency (subject* dep)
{
	dependencies_.push_back(dep);
	dep->attach(this);
	dep->merge_leaves(leaves_);
}

void iobserver::merge_leaves (std::unordered_set<subject*>& src)
{
	src.insert(this->leaves_.begin(), this->leaves_.end());
}

iobserver::iobserver (std::vector<subject*> dependencies)
{
	for (subject* dep : dependencies)
	{
		add_dependency(dep);
	}
}
		
iobserver::~iobserver (void)
{
	for (subject* dep : dependencies_)
	{
		dep->detach(this);
	}
}
		
void iobserver::leaves_collect (std::function<void(subject*)> collector)
{
	for (subject* leaf : leaves_)
	{
		collector(leaf);
	}
}

}

#endif

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
	dep->attach(this, dependencies_.size());
	dependencies_.push_back(dep);
}

iobserver::iobserver (const iobserver& other)
{
	for (subject* dep : other.dependencies_)
	{
		add_dependency(dep);
	}
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

}

#endif

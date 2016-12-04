//
//  iobserver.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <vector>
#include <functional>
#include "subject.hpp"

#pragma once
#ifndef observer_hpp
#define observer_hpp

namespace ccoms
{

// AKA root / intermediate node

class iobserver : public reactive_node
{
	protected:
		//dependencies exposed to inherited to facilitate moving around the graph
		std::vector<subject*> dependencies_;
		
		void add_dependency (subject* dep); // add in order

		void replace_dep (subject* dep, size_t idx)
		{
			dependencies_[idx]->detach(this);
			dependencies_[idx] = dep;
			dep->attach(this, idx);
		}

		void copy (const iobserver& other)
		{
			for (subject* sub : dependencies_)
			{
				sub->detach(this);
			}
			dependencies_.clear();
			for (subject* dep : other.dependencies_)
			{
				add_dependency(dep);
			}
		}
		
		// attach dependencies
		iobserver (const iobserver& other); // copy over dependencies and leaves
		iobserver (std::vector<subject*> dependencies);
		
		virtual bool suicidal (void) { return true; }

	public:
		virtual ~iobserver (void);

		virtual void update (caller_info info, update_message msg = ccoms::update_message()) = 0;
};

}

#endif /* observer_hpp */

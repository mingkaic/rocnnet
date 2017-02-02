//
//  iobserver.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <vector>
#include <functional>
#include "subject_owner.hpp"

#pragma once
#ifndef observer_hpp
#define observer_hpp

namespace ccoms
{

// AKA root / intermediate node

class iobserver : public ireactive_node
{
	protected:
		// dependencies exposed to inherited to facilitate moving around the graph
		// order of subject matters; observer-subject relation is non-unique
		std::vector<subject*> dependencies_; // TODO: move to private

		void add_dependency (subject* dep); // add in order
		void add_dependency (subject_owner* dep_owner)
		{
			add_dependency(dep_owner->caller_.get());
		}
		void kill_dependencies (void)
		{
			for (subject* dep : dependencies_)
			{
				dep->detach(this);
			}
			dependencies_.clear();
		}
		void copy (const iobserver& other); // for assignment ops
		
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

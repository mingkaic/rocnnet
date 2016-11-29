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
	private:
		// remember that once leaf subjects are destroyed, 
		// everyone in this graph including this is destroyed
		// so we don't need to bother with cleaning leaves_
		std::unordered_set<subject*> leaves_;

	protected:
		//dependencies exposed to inherited to facilitate moving around the graph
		std::vector<subject*> dependencies_;
		
		void add_dependency (subject* dep);
		virtual void merge_leaves (std::unordered_set<subject*>& src);
		
		// attach dependencies
		iobserver (const iobserver& other); // copy over dependencies and leaves
		iobserver (std::vector<subject*> dependencies);
		
		virtual bool suicidal (void) { return true; }

	public:
		virtual ~iobserver (void);
		
		void leaves_collect (std::function<void(subject*)> collector);
		virtual void update (update_message msg) = 0;
};

}

#endif /* observer_hpp */

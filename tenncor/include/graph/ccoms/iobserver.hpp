//
//  iobserver.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef observer_hpp
#define observer_hpp

#include <vector>
#include "subject.hpp"

namespace ccoms
{

// AKA root / intermediate node

class iobserver : public ileaf_handler
{
	private:
		// remember that once leaf subjects are destroyed, 
		// everyone in this graph including this is destroyed
		// so we don't need to bother with cleaning leaves_
		std::unordered_set<subject*> leaves_;

	protected:
		//dependencies exposed to inherited to facilitate moving around the graph
		std::vector<subject*> dependencies_;
		
		void add_dependency (ccoms::subject* dep);
		virtual void merge_leaves (std::unordered_set<ccoms::subject*>& src);
		
		// inherited classes may desire empty dependencies 
		// with the option of adding dependencies later
		iobserver (void) {}

	public:
		iobserver (std::vector<ccoms::subject*> dependencies);
		virtual ~iobserver (void);
		
		void leaves_collect (std::function<void(subject*)> collector);
		virtual void update (ccoms::subject* caller) = 0;
};

}

#endif /* observer_hpp */

//
//  node.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef node_hpp
#define node_hpp

#include "observer.hpp"

namespace ccoms {
	
// intermediate communication object, most commonly used
	
class node : public subject, public observer {
	private:
		std::unordered_set<subject*> leaves_;
		
	protected:
		virtual void add_leaves (std::unordered_set<subject*>& src) const {
			src.insert(leaves_.cbegin(), leaves_.cend());
		}
	
	public:
		node (std::vector<subject*> dependencies) :
			observer(dependencies) {
			for (subject* dep : dependencies) {
				dep->add_leaves(leaves_);
			}
		}
		
		virtual void update (void) = 0;
};
	
}

#endif /* node_hpp */

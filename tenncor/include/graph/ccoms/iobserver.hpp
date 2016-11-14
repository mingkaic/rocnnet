//
//  ccoms.hpp
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

namespace ccoms {

// AKA root / intermediate node

class iobserver : public ileaf_handler {
	private:
		// remember that once leaf subjects are destroyed, 
		// then everyone in this graph including this is destroyed
		// so we don't need to bother with cleaning leaves_
		std::unordered_set<subject*> leaves_;

	protected:
		std::vector<subject*> dependencies_;
		
		virtual void merge_leaves (std::unordered_set<ccoms::subject*>& src) {
			src.insert(this->leaves_.begin(), this->leaves_.end());
		}

	public:
		iobserver (std::vector<ccoms::subject*> dependencies) :
			dependencies_(dependencies) {
			for (ccoms::subject* dep : dependencies) {
				dep->attach(this);
			}
			for (ccoms::subject* dep : dependencies) {
				dep->merge_leaves(leaves_);
			}
		}
		
		virtual ~iobserver (void) {
			for (ccoms::subject* dep : dependencies_) {
				dep->detach(this);
			}
		}
		
		void leaves_collect (std::function<void(subject*)> collector) {
			for (ccoms::subject* leaf : leaves_) {
				collector(leaf);
			}
		}
		
		virtual void update (ccoms::subject* caller) = 0;
};

}

#endif /* observer_hpp */

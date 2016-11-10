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

class inode : public subject, public iobserver {
	private:
		std::unordered_set<subject*> leaves_;

	protected:
		virtual void merge_leaves (std::unordered_set<ccoms::subject*>& src) {
			src.insert(leaves_.begin(), leaves_.end());
		}
	
	public:
		inode (std::vector<ccoms::subject*> dependencies) :
			iobserver(dependencies) {
			for (ccoms::subject* dep : dependencies) {
				dep->merge_leaves(leaves_);
			}
		}

		void leaves_collect (std::function<void(ccoms::subject*)> collector) {
			for (ccoms::subject* leaf : leaves_) {
				collector(leaf);
			}
		}
};
	
}

#endif /* node_hpp */

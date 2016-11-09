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
		virtual void merge_leaves (std::unordered_set<subject*>& src) {
			src.insert(leaves_.begin(), leaves_.end());
		}
	
	public:
		inode (std::vector<subject*> dependencies) :
			iobserver(dependencies) {
			for (subject* dep : dependencies) {
				dep->merge_leaves(leaves_);
			}
		}

		void leaves_collect (std::function<void(subject*)> collector) {
			for (subject* leaf : leaves_) {
				collector(leaf);
			}
		}
		
		virtual void update (void) = 0;
};
	
}

#endif /* node_hpp */

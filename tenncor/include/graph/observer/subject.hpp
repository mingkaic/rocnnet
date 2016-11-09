//
//  subject.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef subject_hpp
#define subject_hpp

#include <unordered_set>

namespace ccoms {

class observer;

// AKA leaf

class subject {
	private:
		std::unordered_set<observer*> audience_;
		// raw_data
		
	protected:
		virtual void add_leaves (std::unordered_set<subject*>& src) const {
			src.emplace(this);
		}
		
	public:
		virtual ~subject (void) {
			auto it = audience_.begin();
			while (audience_.end() != it) {
				observer* captive = *it;
				it++;
				delete captive;
			}
		}
	
		void attach (observer* viewer) {
			audience_.emplace(viewer);
		}
		
		void detach (observer* viewer) {
			audience_.erase(viewer);
		}
		
		void notify (void) {
			for (observer viewer : audience_) {
				viewer->update();
			}
		}
};

}

#endif /* subject_hpp */

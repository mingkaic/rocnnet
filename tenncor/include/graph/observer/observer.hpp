//
//  observer.hpp
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

// AKA root

class iobserver {
	protected:
		std::vector<subject*> dependencies_;

	public:
		iobserver (std::vector<subject*> dependencies) :
			dependencies_(dependencies) {
			for (subject* dep : dependencies) {
				dep->attach(this);
			}
		}
		
		virtual ~iobserver (void) {
			for (subject* dep : dependencies_) {
				dep->detach(this);
			}
		}
		
		virtual void update (void) = 0;
};

}

#endif /* observer_hpp */

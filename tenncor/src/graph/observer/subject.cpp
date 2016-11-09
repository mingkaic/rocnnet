//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../../include/graph/observer/subject.hpp"
#include "../../../include/graph/observer/observer.hpp"

#ifdef subject_hpp

namespace ccoms {

subject::~subject (void) {
	auto it = audience_.begin();
	while (audience_.end() != it) {
		iobserver* captive = *it;
		it++;
		delete captive;
	}
}

void subject::notify (void) {
	for (iobserver* viewer : audience_) {
		viewer->update();
	}
}

}

#endif
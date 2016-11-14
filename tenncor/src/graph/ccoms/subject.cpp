//
//  subject.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../../include/graph/ccoms/subject.hpp"
#include "graph/ccoms/iobserver.hpp"

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

void subject::notify (subject* caller) {
	for (iobserver* viewer : audience_) {
		viewer->update(caller);
	}
}

}

#endif
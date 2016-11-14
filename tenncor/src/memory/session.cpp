//
//  session.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-26.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/memory/session.hpp"

#ifdef session_hpp

namespace nnet {

session& session::get_instance (void) {
	static session my_instance;
	return my_instance;
}

std::default_random_engine& session::get_generator (void) {
	session& inst = session::get_instance();
	return inst.get_rand_generator();
}

// void register_obj (ivariable<std::any>& obj) {
//	 registry.emplace(&obj);
// }

}

#endif

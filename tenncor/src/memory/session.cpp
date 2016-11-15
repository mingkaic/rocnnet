//
//  session.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-26.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/memory/session.hpp"

#ifdef session_hpp

namespace nnet
{

session& session::get_instance (void)
{
	static session my_instance;
	return my_instance;
}

bool session::pre_shape_eval (void) {
	session& inst = get_instance();
	return inst.shape_eval;
}

std::default_random_engine& session::get_generator (void)
{
	session& inst = session::get_instance();
	return inst.get_rand_generator();
}

void session::seed_rand_eng (size_t seed)
{
	generator_.seed(seed);
}

std::default_random_engine& session::get_rand_generator (void)
{
	return generator_;
}

void session::enable_shape_eval (void) { shape_eval = true; }

void session::disable_shape_eval (void) { shape_eval = false; }

// void register_obj (ivariable<std::any>& obj)
// {
//	 registry.emplace(&obj);
// }

}

#endif

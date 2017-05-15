//
// Created by Mingkai Chen on 2017-05-11.
//

#include "memory/shared_rand.hpp"

#ifdef TENNCOR_RANDOM_HPP

namespace nnet
{

static std::default_random_engine common_generator(std::time(NULL));

std::default_random_engine& get_generator (void)
{
	return common_generator;
}

void seed_generator (size_t val)
{
	common_generator.seed(val);
}

}

#endif

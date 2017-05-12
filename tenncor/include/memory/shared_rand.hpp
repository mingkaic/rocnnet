//
// Created by Mingkai Chen on 2017-05-11.
//

#pragma once
#ifndef TENNCOR_RANDOM_HPP
#define TENNCOR_RANDOM_HPP

#include <random>
#include <ctime>

namespace nnet
{

std::default_random_engine& get_generator (void);

void seed_generator (size_t val);

}

#endif /* TENNCOR_RANDOM_HPP */

//
// Created by Mingkai Chen on 2016-10-07.
//

#include "rocnnet/nnet.hpp"
#include "gtest/gtest.h"


TEST(DQN, memory_replay) {

}


TEST(DQN, forward) {
	size_t n_out = 3;
	size_t n_hidden = 4;
	std::vector<double> vin = {1, 2, 3, 4, 5};
	std::vector<IN_PAIR> hiddens = {
			// use same sigmoid in static memory once deep copy is established
			IN_PAIR(n_hidden, new nnet::sigmoid<double>()),
			IN_PAIR(n_hidden, new nnet::sigmoid<double>()),
			IN_PAIR(n_out, new nnet::sigmoid<double>()),
	};
	nnet::dq_net net(vin.size(), hiddens);
}


TEST(DQN, training) {

}
//
//  dqn.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>
#include <random>

#include "gd_net.hpp"

#pragma once
#ifndef dqn_hpp
#define dqn_hpp

namespace nnet {

class dq_net {
	private:
		// argument memorization
		size_t n_observations; // input
		size_t n_actions; // output

		ml_perceptron* q_net;
		ml_perceptron* target_net;

		double rand_action_prob;
		size_t exploration_period;
		size_t store_interval;
		size_t train_interval;
		size_t mini_batch_size;
		double discount_rate;
		size_t max_exp;
		double update_rate;

		// state
		struct exp_batch {
			std::vector<double> observation;
			size_t action_idx;
			double reward;
			std::vector<double> new_observation;
			exp_batch(
				std::vector<double> observation,
				size_t action_idx,
				double reward,
				std::vector<double> new_observation) :
					observation(observation),
					action_idx(action_idx),
					reward(reward),
					new_observation(new_observation) {}
		};

		std::vector<exp_batch> experiences;
		size_t actions_executed;
		size_t iteration;
		size_t n_store_called;
		size_t n_train_called;

		// intermediate variables
		std::unordered_set<ivariable<double>*> ownership;
		// fanins
		placeholder<double>* observation;
		placeholder<double>* next_observation;
		placeholder<double>* next_observation_mask;
		placeholder<double>* rewards;
		placeholder<double>* action_mask;
		// fanouts
		ivariable<double>* predicted_actions;
		ivariable<double>* prediction_error;

		double get_random(void) {
			static std::default_random_engine generator;
			static std::uniform_real_distribution<double> explore;

			return explore(generator);
		}

		std::vector<exp_batch> get_sample (void) {
			std::vector<size_t> indices(experiences.size());
			std::iota(indices.begin(), indices.end(), 0);
			std::random_shuffle(indices.begin(), indices.end());
			std::vector<exp_batch> res;
			for (size_t idx : indices) {
				res.push_back(experiences[idx]);
			}
			return res;
		}

		void clear (void) {
			delete q_net;
			delete target_net;

			delete observation;
			delete next_observation;
			delete next_observation_mask;
			delete rewards;
			delete action_mask;
			delete predicted_actions;
			delete prediction_error;

			for (ivariable<double>* v : ownership) {
				delete v;
			}
			ownership.clear();
		}

		double linear_annealing (double initial_prob) const {
			if (actions_executed >= exploration_period) {
				return rand_action_prob;
			}
			return initial_prob - actions_executed *
				(initial_prob - rand_action_prob)
				/ exploration_period;
		}

	public:
		dq_net (
			size_t n_input,
			std::vector<IN_PAIR> hiddens,
			size_t train_interval = 5,
			double rand_action_prob = 0.05,
			double discount_rate = 0.95,
			double update_rate = 0.01,
			// memory parameters
			size_t store_interval = 5,
			size_t mini_batch_size = 32,
			size_t max_exp = 30000);

		virtual ~dq_net (void) { clear(); }

		std::vector<double> operator () (std::vector<double>& input);

		// memory replay
		void store (
			std::vector<double> observation,
			size_t action_idx,
			double reward,
			std::vector<double> new_obs);

		void train (std::vector<std::vector<double> > train_batch);

//		// persistence
//		void save (void); // implement
//		void restore (void); // implement
};

}

#endif /* dqn_hpp */

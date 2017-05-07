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

#include "mlp.hpp"
#include "utils/gd_utils.hpp"

#pragma once
#ifndef dqn_hpp
#define dqn_hpp

namespace rocnnet
{

struct dqn_param
{
	size_t train_interval_ = 5;
	double rand_action_prob_ = 0.05;
	double discount_rate_ = 0.95;
	double update_rate_ = 0.01;
	double exploration_period_ = 1000;
	// memory parameters
	size_t store_interval_ = 5;
	size_t mini_batch_size_ = 32;
	size_t max_exp_ = 30000;
};

class dq_net
{
public:
	dq_net (size_t n_input, std::vector<IN_PAIR> hiddens,
		nnet::gd_updater<double>& updater,
		dqn_param param = dqn_param(),
		std::string scope = "DQN");

	~dq_net (void);

	dq_net (const dq_net& other, std::string scope);

	dq_net (dq_net&& other, std::string scope);

	dq_net& operator = (const dq_net& other);

	dq_net& operator = (dq_net&& other);

	std::vector<double> action (std::vector<double>& input);

	void store (std::vector<double> observation, size_t action_idx,
		double reward, std::vector<double> new_obs);

	void train (void);

	void initialize (std::string serialname = "");

	bool save (std::string fname) const;

	// feel free to seed it
	std::default_random_engine generator_;

private:
	// experience replay
	struct exp_batch
	{
		std::vector<double> observation_;
		size_t action_idx_;
		double reward_;
		std::vector<double> new_observation_;
	};

	// delete everything
	void tear_down (void);

	// copy setup
	void copy_helper (const dq_net& other, std::string scope);

	// move setup
	void move_helper (dq_net&& other, std::string scope);

	// set up the output nodes
	void variable_setup (void);

	double linear_annealing (double initial_prob) const;

	double get_random (void);

	std::vector<exp_batch> random_sample (void);

	// source network
	ml_perceptron* source_qnet_;

	// target network
	ml_perceptron* target_qnet_;

	// === forward computation ===
	// fanin: shape <ninput, batchsize>
	nnet::placeholder<double>* input_ = nullptr;

	// fanout: shape <noutput, batchsize>
	nnet::varptr<double> output_ = nullptr;

	// fanout: scalar shape
	nnet::varptr<double> best_output = nullptr;

	// === prediction computation ===
	// train_fanin: shape <ninput, batchsize>
	nnet::placeholder<double>* next_input_ = nullptr;

	// train mask: shape <batchsize>
	nnet::placeholder<double>* next_output_mask_ = nullptr;

	// train fanout: shape <noutput, batchsize>
	nnet::varptr<double> next_output_ = nullptr;

	// reward associated with next_output_: shape <batchsize>
	nnet::placeholder<double>* reward_ = nullptr;

	// future reward calculated from reward history: <1, batchsize>
	nnet::varptr<double> future_reward_ = nullptr;

	// === q-value computation ===
	// weight output to get overall score: shape <noutput, batchsize>
	nnet::placeholder<double>* output_mask_ = nullptr;

	// overall score: shape <noutput>
	nnet::varptr<double> score_ = nullptr;

	// future error that we want to minimize: scalar shape
	nnet::iconnector<double>* error_ = nullptr;

	// === updates && optimizer ===
	// update source network (this) using updater
	std::vector<nnet::variable_updater<double> > source_updates_;

	// update target network (target_qnet) from source weights
	std::vector<nnet::variable_updater<double> > target_updates_;

	// optimizer
	nnet::gd_updater<double>* updater_ = nullptr;

	// === scalar parameters ===
	// argument memorization
	size_t n_input_; // input size

	size_t n_output_; // output size

	// training parameters
	dqn_param params_;

	// states
	size_t actions_executed_ = 0;

	size_t iteration_ = 0;

	size_t n_store_called_ = 0;

	size_t n_train_called_ = 0;

	std::uniform_real_distribution<double> explore_;

	std::vector<exp_batch> experiences_;
};

}

#endif /* dqn_hpp */
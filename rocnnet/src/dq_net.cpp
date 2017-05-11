//
//  dqn.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "dq_net.hpp"

#ifdef dqn_hpp

namespace rocnnet
{

dq_net::dq_net (size_t n_input, std::vector<IN_PAIR> hiddens,
	nnet::gd_updater<double>& updater,
	dqn_param param, std::string scope) :
n_input_(n_input), params_(param),
updater_(updater.clone())
{
	source_qnet_ = new ml_perceptron(n_input, hiddens, "source_"+scope);
	target_qnet_ = source_qnet_->clone("target_"+scope);
	n_output_ = hiddens.back().first;
	input_ = new nnet::placeholder<double>(std::vector<size_t>{n_input, 1}, "input");
	output_mask_ = new nnet::placeholder<double>(std::vector<size_t>{n_output_, 0}, "output_mask");

	train_input_ = new nnet::placeholder<double>(std::vector<size_t>{n_input, 0}, "input");
	next_input_ = new nnet::placeholder<double>(std::vector<size_t>{n_input, 0}, "next_input");
	next_output_mask_ = new nnet::placeholder<double>(std::vector<size_t>{0}, "next_output_mask");
	reward_ = new nnet::placeholder<double>(std::vector<size_t>{0}, "next_reward");
	variable_setup();
}

dq_net::~dq_net (void)
{
	tear_down();
}

dq_net::dq_net (const dq_net& other, std::string scope)
{
	copy_helper(other, scope);
}

dq_net::dq_net (dq_net&& other, std::string scope)
{
	move_helper(std::move(other), scope);
}

dq_net& dq_net::operator = (const dq_net& other)
{
	if (this != &other)
	{
		copy_helper(other, "");
	}
	return *this;
}

dq_net& dq_net::operator = (dq_net&& other)
{
	if (this != &other)
	{
		move_helper(std::move(other), "");
	}
	return *this;
}

double dq_net::action (std::vector<double>& input)
{
	actions_executed_++; // book keep
	double exploration = linear_annealing(1.0);
	// perform random exploration action
	if (get_random() < exploration)
	{
		return std::floor(get_random() * n_output_);
	}
	*input_ = input;
	return nnet::expose<double>(best_output_)[0];
}
	
std::vector<double> dq_net::direct_out (std::vector<double>& input)
{
	actions_executed_++; // book keep
	double exploration = linear_annealing(1.0);
	// perform random exploration action
	if (get_random() < exploration)
	{
		std::vector<double> act_score(n_output_); 
	    std::generate(act_score.begin(), act_score.end(), [this](){ return get_random(); }); 
	    return act_score; 
	}
	*input_ = input;
	return nnet::expose<double>(output_);
}

void dq_net::store (std::vector<double> observation, size_t action_idx,
	double reward, std::vector<double> new_obs)
{
	if (0 == n_store_called_ % params_.store_interval_)
	{
		experiences_.push_back(exp_batch{observation, action_idx, reward, new_obs});
		if (experiences_.size() > params_.max_exp_)
		{
			experiences_.front() = std::move(experiences_.back());
			experiences_.pop_back();
		}
	}
	n_store_called_++;
}

void dq_net::train (void)
{
	// extract mini_batch from buffer and backpropagate
	if (0 == (n_train_called_ % params_.train_interval_))
	{
		if (experiences_.size() < params_.mini_batch_size_) return;

		std::vector<exp_batch> samples = random_sample();

		// batch data process
		std::vector<double> states; // <ninput, batchsize>
		std::vector<double> new_states; // <ninput, batchsize>
		std::vector<double> action_mask; // <noutput, batchsize>
		std::vector<double> new_states_mask; // <batchsize>
		std::vector<double> rewards; // <batchsize>

		for (size_t i = 0, n = samples.size(); i < n; i++)
		{
			exp_batch batch = samples[i];
			states.insert(states.end(), batch.observation_.begin(), batch.observation_.end());
			{
				std::vector<double> local_act_mask(n_output_, 0);
				local_act_mask[batch.action_idx_] = 1.0;
				action_mask.insert(action_mask.end(), local_act_mask.begin(), local_act_mask.end());
			}
			rewards.push_back(batch.reward_);
			if (batch.new_observation_.empty())
			{
				new_states.insert(new_states.end(), n_input_, 0);
				new_states_mask.push_back(0);
			}
			else
			{
				new_states.insert(new_states.end(), batch.new_observation_.begin(), batch.new_observation_.end());
				new_states_mask.push_back(1);
			}
		}

		// enter processed batch data
		*train_input_ = states;
		*output_mask_ = action_mask;
		*next_input_ = new_states;
		*next_output_mask_ = new_states_mask;
		*reward_ = rewards;

		error_->update_status(true); // freeze
		// update source
		for (auto& trainer : source_updates_)
		{
			trainer();
		}
		// update target
		for (auto& trainer : target_updates_)
		{
			trainer();
		}
		error_->update_status(false); // update again
		iteration_++;
	}
	n_train_called_++;
}

void dq_net::initialize (std::string serialname)
{
	source_qnet_->initialize(serialname);
	target_qnet_->initialize(serialname);
}

bool dq_net::save (std::string fname) const
{
	bool source_save = source_qnet_->save(fname);
	bool target_save = target_qnet_->save(fname);
	return source_save && target_save;
}

void dq_net::tear_down (void)
{
	// cascade delete all leaf nodes 
	// (qnet for the variables, then local placeholders)
	if (source_qnet_) delete source_qnet_;
	if (target_qnet_) delete target_qnet_;
	
	if (input_) delete input_;
	if (train_input_) delete train_input_;
	if (output_mask_) delete output_mask_;
	if (next_input_) delete next_input_;
	if (next_output_mask_) delete next_output_mask_;
	if (reward_) delete reward_;
	
	source_qnet_ = nullptr;
	target_qnet_ = nullptr;
	
	// nullify leaf placeholders
	input_ = nullptr;
	train_input_ = nullptr;
	output_mask_ = nullptr;
	next_input_ = nullptr;
	next_output_mask_ = nullptr;
	reward_ = nullptr;
	
	// nullify graph roots
	output_ = nullptr;
	best_output_ = nullptr;
	train_output_ = nullptr;
	next_output_ = nullptr;
	future_reward_ = nullptr;
	score_ = nullptr;
	error_ = nullptr;
	
	// clear updates
	source_updates_.clear();
	target_updates_.clear();
	updater_ = nullptr;
}

void dq_net::copy_helper (const dq_net& other, std::string scope)
{
	// copy over parameters
	n_input_ = other.n_input_;
	n_output_ = other.n_output_;
	params_  = other.params_;
	actions_executed_ = other.actions_executed_;
	iteration_ = other.iteration_;
	n_store_called_ = other.n_store_called_;
	n_train_called_ = other.n_train_called_;
	explore_ = other.explore_;
	generator_ = other.generator_;

	// copy over actors
	tear_down();
	updater_ = other.updater_->clone();
	updater_->clear_ignore();
	
	source_qnet_ = other.source_qnet_->clone("source_"+scope);
	target_qnet_ = other.target_qnet_->clone("target_"+scope);
	input_ = other.input_->clone();
	
	train_input_ = other.train_input_->clone();
	output_mask_ = other.output_mask_->clone();
	next_input_ = other.next_input_->clone();
	next_output_mask_ = other.next_output_mask_->clone();
	reward_ = other.reward_->clone();
	variable_setup();
}

void dq_net::move_helper (dq_net&& other, std::string scope)
{
	// move parameters
	n_input_ = std::move(other.n_input_);
	n_output_ = std::move(other.n_output_);
	params_  = std::move(other.params_);
	actions_executed_ = std::move(other.actions_executed_);
	iteration_ = std::move(other.iteration_);
	n_store_called_ = std::move(other.n_store_called_);
	n_train_called_ = std::move(other.n_train_called_);
	explore_ = std::move(other.explore_);
	generator_ = std::move(other.generator_);

	// move over actors
	tear_down();
	updater_ = other.updater_->move();
	updater_->clear_ignore();
	
	source_qnet_ = other.source_qnet_->move("source_"+scope);
	target_qnet_ = other.target_qnet_->move("target_"+scope);
	input_ = other.input_->move();
	
	train_input_ = other.train_input_->move();
	output_mask_ = other.output_mask_->move();
	next_input_ = other.next_input_->clone();
	next_output_mask_ = other.next_output_mask_->move();
	reward_ = other.reward_->move();
	variable_setup();
}

void dq_net::variable_setup (void)
{
	// output computation
	output_ = (*source_qnet_)(input_);
	output_->set_label("output");
	best_output_ = nnet::arg_max(output_, 0);

	// training input/output
	train_output_ = (*source_qnet_)(train_input_);
	train_output_->set_label("train_output");

	// predict future reward
	next_output_ = (*target_qnet_)(next_input_);
	next_output_->set_label("next_output");
	nnet::varptr<double> target_values = nnet::varptr<double>(next_output_mask_) *
		nnet::reduce_max(nnet::varptr<double>(next_output_), 0);
	future_reward_ = nnet::varptr<double>(reward_) + params_.discount_rate_ * target_values; // reward for each instance in batch
	future_reward_->set_label("future_reward");

	// predict future error
	nnet::varptr<double> masked_output_score = nnet::reduce_sum(
		nnet::varptr<double>(train_output_) * nnet::varptr<double>(output_mask_), 0);
	nnet::varptr<double> diff = masked_output_score - future_reward_;
	nnet::varptr<double> error = nnet::reduce_mean(diff * diff);
	error_ = static_cast<nnet::iconnector<double>*>(error.get());
	error_->set_label("error");

	// updates for source network
	std::unordered_set<nnet::variable<double>*> biases;
	{
		std::vector<WB_PAIR> wbs = source_qnet_->get_variables();
		for (WB_PAIR& wb : wbs)
		{
			biases.emplace(wb.second);
		}
	}
	updater_->ignore_subtree(next_output_);
	source_updates_ = updater_->calculate(error_,
	[biases](nnet::varptr<double> grad, nnet::variable<double>* leaf)
	{
		if (biases.end() != biases.find(leaf))
		{
			// average the batches
			grad = nnet::reduce_mean(grad, {1});
		}
		return nnet::clip_norm(grad, 5.0);
	});

	// updates for target network
	std::vector<WB_PAIR> target_vars = target_qnet_->get_variables();
	std::vector<WB_PAIR> source_vars = source_qnet_->get_variables();
	size_t nvars = target_vars.size();
	assert(source_vars.size() == nvars); // must be identical, since they're copies
	for (size_t i = 0; i < nvars; i++)
	{
		// this is equivalent to target = (1-alpha) * target + alpha * source
		nnet::varptr<double> tarweight = target_vars[i].first;
		nnet::varptr<double> srcweight = source_vars[i].first;
		target_updates_.push_back(target_vars[i].first->assign_sub(params_.update_rate_ * (tarweight - srcweight)));

		nnet::varptr<double> tarbias = target_vars[i].second;
		nnet::varptr<double> srcbias = source_vars[i].second;
		target_updates_.push_back(target_vars[i].second->assign_sub(params_.update_rate_ * (tarbias - srcbias)));
	}
}

double dq_net::linear_annealing (double initial_prob) const
{
	if (actions_executed_ >= params_.exploration_period_)
		return params_.rand_action_prob_;
	return initial_prob - actions_executed_ * (initial_prob - params_.rand_action_prob_)
		/ params_.exploration_period_;
}

double dq_net::get_random(void)
{
	return explore_(generator_);
}

std::vector<dq_net::exp_batch> dq_net::random_sample (void)
{
	std::vector<size_t> indices(experiences_.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::random_shuffle(indices.begin(), indices.end());
	std::vector<dq_net::exp_batch> res;
	for (size_t i = 0; i < params_.mini_batch_size_; i++)
	{
		size_t idx = indices[i];
		res.push_back(experiences_[idx]);
	}
	return res;
}

}

#endif

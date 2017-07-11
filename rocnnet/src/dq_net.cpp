//
//  dqn.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "dq_net.hpp"

#ifdef ROCNNET_DQN_HPP

#include "utils/utils.hpp"

namespace rocnnet
{

dq_net::dq_net (ml_perceptron* brain,
	nnet::gd_updater& updater,
	dqn_param param, std::string scope) :
params_(param),
scope_(scope),
updater_(updater.clone())
{
	source_qnet_ = brain;
	target_qnet_ = source_qnet_->clone("target_"+scope);

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
		return std::floor(get_random() * source_qnet_->get_noutput());
	}
	*input_ = input;
	std::vector<double> best = nnet::expose<double>(best_output_);
	assert(false == best.empty());
	return best[0];
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
				std::vector<double> local_act_mask(source_qnet_->get_noutput(), 0);
				local_act_mask[batch.action_idx_] = 1.0;
				action_mask.insert(action_mask.end(), local_act_mask.begin(), local_act_mask.end());
			}
			rewards.push_back(batch.reward_);
			if (batch.new_observation_.empty())
			{
				new_states.insert(new_states.end(), source_qnet_->get_ninput(), 0);
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

		prediction_error_->freeze_status(true); // freeze
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
		prediction_error_->freeze_status(false); // update again
		iteration_++;
	}
	n_train_called_++;
}

void dq_net::initialize (std::string serialname, std::string readscope)
{
	if (readscope.empty()) readscope = scope_;
	source_qnet_->initialize(serialname, readscope);
	target_qnet_->initialize(serialname, "target_" + readscope);
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
	if (source_qnet_)
		delete source_qnet_;
	if (target_qnet_)
		delete target_qnet_;
	
	if (input_) delete input_;
	if (train_input_) delete train_input_;
	if (output_mask_) delete output_mask_;
	if (next_input_) delete next_input_;
	if (next_output_mask_) delete next_output_mask_;
	if (reward_) delete reward_;
	
	if (updater_) delete updater_;
	
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
	prediction_error_ = nullptr;
	
	// clear updates
	source_updates_.clear();
	target_updates_.clear();
	updater_ = nullptr;
}

void dq_net::copy_helper (const dq_net& other, std::string scope)
{
	scope_ = other.scope_;

	// copy over parameters
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
	
	source_qnet_ = other.source_qnet_->clone(scope);
	target_qnet_ = other.target_qnet_->clone("target_"+scope);

	variable_setup();
}

void dq_net::move_helper (dq_net&& other, std::string scope)
{
	scope_ = std::move(other.scope_);

	// move parameters
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
	
	source_qnet_ = other.source_qnet_->move(scope);
	target_qnet_ = other.target_qnet_->move("target_"+scope);

	variable_setup();
}

void dq_net::variable_setup (void)
{
	input_ = new nnet::placeholder<double>(std::vector<size_t>{source_qnet_->get_ninput(), 1}, "observation");
	train_input_ = new nnet::placeholder<double>(std::vector<size_t>{source_qnet_->get_ninput(), 0}, "train_observation");
	next_input_ = new nnet::placeholder<double>(std::vector<size_t>{source_qnet_->get_ninput(), 0}, "next_observation");
	next_output_mask_ = new nnet::placeholder<double>(std::vector<size_t>{0}, "next_observation_mask");
	reward_ = new nnet::placeholder<double>(std::vector<size_t>{0}, "rewards");
	output_mask_ = new nnet::placeholder<double>(std::vector<size_t>{source_qnet_->get_noutput(), 0}, "action_mask");

	// forward action score computation
	output_ = nnet::identity((*source_qnet_)(input_));
	output_->set_label("action_scores");
	best_output_ = nnet::arg_max(output_, 0);

	train_output_ = nnet::identity((*source_qnet_)(train_input_));
	train_output_->set_label("train_action_scores");

	// predicting target future rewards
	next_output_ = nnet::identity((*target_qnet_)(next_input_));
	next_output_->set_label("next_action_scores");

	nnet::varptr<double> target_values =
		nnet::reduce_max(nnet::varptr<double>(next_output_), 0) *
		nnet::varptr<double>(next_output_mask_);
	future_reward_ = nnet::varptr<double>(reward_) + params_.discount_rate_ * target_values; // reward for each instance in batch

	// prediction error
	nnet::varptr<double> masked_output_score = nnet::reduce_sum(
		nnet::varptr<double>(train_output_) * nnet::varptr<double>(output_mask_), 0);
	nnet::varptr<double> temp_diff = masked_output_score - future_reward_;
	nnet::varptr<double> error = nnet::reduce_mean(nnet::pow(temp_diff, 2));
	prediction_error_ = static_cast<nnet::iconnector<double>*>(error.get());
	prediction_error_->set_label("error");

	// updates for source network
	updater_->ignore_subtree(next_output_);
	source_updates_ = updater_->calculate(prediction_error_,
	[](nnet::varptr<double> grad, nnet::variable<double>* leaf)
	{
		grad = nnet::clip_norm(grad, 5.0);
		return grad;
	});

	// update target network
	std::vector<WB_PAIR> target_vars = target_qnet_->get_variables();
	std::vector<WB_PAIR> source_vars = source_qnet_->get_variables();
	size_t nvars = target_vars.size();
	assert(source_vars.size() == nvars); // must be identical, since they're copies
	for (size_t i = 0; i < nvars; i++)
	{
		// this is equivalent to target = (1-alpha) * target + alpha * source
		nnet::variable<double>* tweight;
		nnet::varptr<double> tarweight = tweight = target_vars[i].first;
		nnet::varptr<double> srcweight = source_vars[i].first;
		target_updates_.push_back(tweight->assign_sub(params_.target_update_rate_ * (tarweight - srcweight)));

		nnet::variable<double>* tbias;
		nnet::varptr<double> tarbias = tbias = target_vars[i].second;
		nnet::varptr<double> srcbias = source_vars[i].second;
		target_updates_.push_back(tbias->assign_sub(params_.target_update_rate_ * (tarbias - srcbias)));
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
	return explore_(nnutils::get_generator());
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

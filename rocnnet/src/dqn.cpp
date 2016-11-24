//
//  dqn.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../include/dqn.hpp"

#ifdef dqn_hpp

namespace nnet {

// ===============================
// ACTION AND TRAINING VARIABLES!
// ===============================
void dq_net::variable_setup (void) {
	// ACTION SCORE COMPUTATION
	// ===============================
	nnet::varptr<double> action_scores = (*target_net)(observation);
	predicted_actions = compress<double>(action_scores, 1,
		[](const std::vector<double>& v) {
			// max arg index
			size_t big_idx = 0;
			for (size_t i = 1; i < v.size(); i++) {
				if (v[big_idx] < v[i]) {
					big_idx = i;
				}
			}
			return big_idx;
		});

	// PREDICT FUTURE REWARDS
	// ===============================
	nnet::ivariable<double>* next_action_scores = (*target_net)(next_observation);
	// reduce max
	nnet::varptr<double> target_values = compress<double>(next_action_scores, 1,
	[](const std::vector<double>& v) {
		double big;
		auto it = v.begin();
		big = *it;
		for (it++; v.end() != it; it++) {
			big = big > *it ? big : *it;
		}
		return big;
	});
	// future rewards = rewards + discount * target action
	nnet::varptr<double> future_rewards = nnet::varptr<double>(rewards) + (discount_rate * target_values);

	// PREDICT ERROR
	// ===============================
	nnet::varptr<double> masked_action_score = // reduce sum
	compress<double>(action_scores * varptr<double>(action_mask), 1,
	[](const std::vector<double>& v) {
		double accum;
		for (double d : v) {
			accum += d;
		}
		return accum;
	});
	nnet::varptr<double> diff = masked_action_score - future_rewards;
	prediction_error = compress<double>(diff * diff); // reduce mean
	// action score is used for prediction error, no need to update twice, 
	// especially if we're asynchronously updating (extra conflicts)
	train_op_->ignore(next_action_scores);
	// approach minima in error manifold 
	// sets root, freeze, then manipulate.
	// evaluate gradient of prediction_error (minimize it)
	train_op_->set_manipulate(prediction_error,
	[](ivariable<double>* key,ivariable<double>*& value)
	{
		// manipulate gradient by clipping to reduce outliers
		if (nullptr != value) {
			value = clip_norm<double>(value, 5);
		}
		return true;
	});

	// UPDATE TARGET NETWORK
	// ===============================
	std::vector<WB_PAIR> q_net_var = q_net->get_variables();
	std::vector<WB_PAIR> target_q_net_var = q_net->get_variables();
	for (size_t i = 0; i < q_net_var.size(); i++) {
		nnet::varptr<double> dwt = update_rate * (varptr<double>(q_net_var[i].first) - varptr<double>(target_q_net_var[i].first));
		nnet::varptr<double> dbt = update_rate * (varptr<double>(q_net_var[i].second) - varptr<double>(target_q_net_var[i].second));

		WB_PAIR wb = target_q_net_var[i];
		iexecutor<double>* w_evok = new assign_sub<double>(wb.first, dwt);
		iexecutor<double>* b_evok = new assign_sub<double>(wb.second, dbt);
		net_train.add(w_evok);
		net_train.add(b_evok);
	}
}

double dq_net::get_random(void) {
	static std::default_random_engine generator;
	static std::uniform_real_distribution<double> explore;

	return explore(generator);
}

std::vector<dq_net::exp_batch> dq_net::get_sample (void) {
	std::vector<size_t> indices(experiences.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::random_shuffle(indices.begin(), indices.end());
	std::vector<dq_net::exp_batch> res;
	for (size_t idx : indices) {
		res.push_back(experiences[idx]);
	}
	return res;
}

double dq_net::linear_annealing (double initial_prob) const {
	if (actions_executed >= exploration_period) {
		return rand_action_prob;
	}
	return initial_prob - actions_executed *
						  (initial_prob - rand_action_prob)
						  / exploration_period;
}

dq_net::dq_net (
	size_t n_input,
	std::vector<IN_PAIR> hiddens,
	nnet::ioptimizer<double>* optimizer,
	size_t train_interval,
	double rand_action_prob,
	double discount_rate,
	double update_rate,
	// memory parameters
	size_t store_interval,
	size_t mini_batch_size,
	size_t max_exp) :
	// state parameters
	n_observations(n_input),
	rand_action_prob(rand_action_prob),
	store_interval(store_interval),
	train_interval(train_interval),
	mini_batch_size(mini_batch_size),
	discount_rate(discount_rate),
	max_exp(max_exp),
	update_rate(update_rate),
	// internal states
	actions_executed(0),
	iteration(0),
	n_store_called(0),
	n_train_called(0)
{

	session& sess = session::get_instance();

	IN_PAIR lastpair = *(hiddens.rbegin());
	n_actions = lastpair.first;

	q_net = new ml_perceptron(n_input, hiddens, "q_network");

	// fanin setup
	target_net = q_net->clone("target_network");
	tensorshape in_shape = std::vector<size_t>{n_input};
	observation = new placeholder<double>(in_shape, "observation");
	next_observation = new placeholder<double>(in_shape, "next_observation");
	// mask and reward shape depends on batch size
	next_observation_mask = new placeholder<double>(std::vector<size_t>{n_observations, 0}, "new_observation_mask");
	rewards = new placeholder<double>(std::vector<size_t>{0}, "rewards");
	action_mask = new placeholder<double>(std::vector<size_t>{n_actions, 0}, "action_mask");

	train_op_ = optimizer; // clone?
	variable_setup();

	sess.initialize_all<double>();

	// other initialization such as saver
}

std::vector<double> dq_net::operator () (std::vector<double>& observations) {
	// action is based on 1 set of observations
	actions_executed++; // book keep
	double exploration = linear_annealing(1.0);
	// perform random exploration action
	if (get_random() < exploration) {
		std::vector<double> act_score(n_actions);
		std::generate(act_score.begin(), act_score.end(), [this](){ return get_random(); });
		return act_score;
	}
	// plug in data
	*observation = observations;
	return nnet::expose<double>(predicted_actions);
}

void dq_net::store (std::vector<double> observation,
					size_t action_idx, double reward,
					std::vector<double> new_obs) {
	if (0 == n_store_called % store_interval) {
		experiences.push_back(exp_batch(observation, action_idx, reward, new_obs));
		if (experiences.size() > max_exp) {
			experiences.erase(experiences.begin()); // not that efficient :/ meh
		}
	}
	n_store_called++;
}

void dq_net::train (std::vector<std::vector<double> > train_batch) {
	// extract mini_batch from buffer and backpropagate
	if (0 == n_train_called % train_interval &&
		experiences.size() >= mini_batch_size) {
		std::vector<exp_batch> samples = get_sample();

		// process samples
		std::vector<double> states; // n_observation by mini_batch_size
		std::vector<double> new_states; // n_observation by mini_batch_size
		std::vector<double> action_mask; // n_action by mini_batch_size
		std::vector<double> new_states_mask; // mini_batch_size
		std::vector<double> rewards; // mini_batch_size

		for (size_t i = 0; i < experiences.size(); i++) {
			exp_batch batch = experiences[i];
			// states
			states.insert(states.end(),
				batch.observation.begin(),
				batch.observation.end());
			// action_mask
			std::vector<double> local_act_mask(n_actions, 0);
			local_act_mask[batch.action_idx] = 1.0;
			action_mask.insert(action_mask.end(),
				local_act_mask.begin(),
				local_act_mask.end());
			// rewards
			rewards.push_back(batch.reward);
			// new_states and new_states_mask
			if (batch.new_observation.empty()) {
				new_states.insert(new_states.end(), n_observations, 0);
				new_states_mask.push_back(0);
			} else {
				new_states.insert(
					new_states.end(),
					batch.new_observation.begin(),
					batch.new_observation.end());
				new_states_mask.push_back(1);
			}
		}

		(*observation) = states;
		(*next_observation) = new_states;
		(*next_observation_mask) = new_states_mask;
		(*this->action_mask) = action_mask;
		(*this->rewards) = rewards;

		// record predictions if necessary
//		prediction_error->eval(); // cost

		// weight training
		train_op_->execute();

		// update q nets
		net_train.execute();

		iteration++;
	}
	n_train_called++;
}

}

#endif

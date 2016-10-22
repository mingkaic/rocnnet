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
		n_train_called(0) {

	session& sess = session::get_instance();

	IN_PAIR lastpair = *(hiddens.rbegin());
	n_actions = lastpair.first;

	q_net = new ml_perceptron(n_input, hiddens, "q_network");

	// fanin setup
	target_net = q_net->clone("target_network");
	tensor_shape in_shape = std::vector<size_t>{n_input};
	observation = std::make_shared<placeholder<double> >(in_shape, "observation");
	next_observation = std::make_shared<placeholder<double> >(in_shape, "next_observation");
	// mask and reward shape depends on batch size
	next_observation_mask = std::make_shared<placeholder<double> >(
			std::vector<size_t>{n_observations, 0}, "new_observation_mask");
	rewards = std::make_shared<placeholder<double> >(std::vector<size_t>{0}, "rewards");
	action_mask = std::make_shared<placeholder<double> >(std::vector<size_t>{n_actions, 0}, "action_mask");

	// ===============================
	// ACTION AND TRAINING VARIABLES!
	// ===============================

	// ACTION SCORE COMPUTATION
	// ===============================
	VAR_PTR<double> action_scores = (*target_net)(observation);
	predicted_actions = // max arg index
		std::make_shared<compress<double> >(action_scores, 1, [](const std::vector<double>& v) {
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
	VAR_PTR<double> next_action_scores = (*target_net)(next_observation);
	VAR_PTR<double> target_values = // reduce max
		std::make_shared<compress<double> >(next_action_scores, 1,
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
	VAR_PTR<double> mulop = std::make_shared<mul<double> >(discount_rate, target_values);
	VAR_PTR<double> future_rewards = std::make_shared<add<double> >(rewards, mulop);

	// PREDICT ERROR
	// ===============================
	VAR_PTR<double> inter_mul = std::make_shared<mul<double> >(action_scores, action_mask);
	VAR_PTR<double> masked_action_score = // reduce sum
		std::make_shared<compress<double> >(inter_mul, 1,
			[](const std::vector<double>& v) {
			double accum;
			for (double d : v) {
				accum += d;
			}
			return accum;
		});
	VAR_PTR<double> tempdiff = std::make_shared<sub<double> >(masked_action_score, future_rewards);
	VAR_PTR<double> sqrdiff = std::make_shared<mul<double> >(tempdiff, tempdiff);

	prediction_error = std::make_shared<compress<double> >(sqrdiff); // reduce mean
	// minimize error
	VAR_PTR<double> grads = std::make_shared<gradient<double> >(prediction_error);

	// TODO: do something with gradient (update by gradient descent?) attach update operation as an output

	sess.initialize_all<double>();

	// other initialization such as saver
}

std::vector<double> dq_net::operator () (std::vector<double>& observations) {
	// action is based on 1 set of observations
	actions_executed++;
	double exploration = linear_annealing(1.0);

	if (get_random() < exploration) {
		std::vector<double> act_score(n_actions);
		std::generate(act_score.begin(), act_score.end(), [this](){ return get_random(); });

		return act_score;
	}

	(*observation) = observations;
	expose<double> out(predicted_actions);
	return out.get_raw();
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

		prediction_error->eval(); // cost
		// do something to update networks

		iteration++;
	}
	n_train_called++;
}

}

#endif

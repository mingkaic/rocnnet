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

	// ===============================
	// ACTION AND TRAINING VARIABLES!
	// ===============================
	tensor_shape in_shape = std::vector<size_t>{n_input};
	target_net = q_net->clone("target_network");

	// ACTION SCORE COMPUTATION
	// ===============================
	observation = new placeholder<double>(in_shape, "observation");
	ivariable<double>& action_scores = (*target_net)(*observation);
	predicted_actions = // max arg index
		new compress<double>(action_scores, 1, [](const std::vector<double>& v) {
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
	next_observation = new placeholder<double>(in_shape, "next_observation");
	ivariable<double>& next_action_scores = (*target_net)(*next_observation);
	// unknown shapes
	// mask and reward shape depends on batch size
	next_observation_mask = new placeholder<double>(
		std::vector<size_t>{n_observations, mini_batch_size}, "new_observation_mask");
	rewards = new placeholder<double>(std::vector<size_t>{mini_batch_size}, "rewards");

	ivariable<double>* target_values = // reduce max
		new compress<double>(next_action_scores, 1, [](const std::vector<double>& v) {
			double big;
			auto it = v.begin();
			big = *it;
			for (it++; v.end() != it; it++) {
				big = big > *it ? big : *it;
			}
			return big;
		});
	// future rewards = rewards + discount * target action
	ivariable<double>* mulop = new mul<double>(discount_rate, *target_values);
	ivariable<double>* future_rewards = new add<double>(*rewards, *mulop);
	ownership.emplace(target_values);
	ownership.emplace(mulop);
	ownership.emplace(future_rewards);

	// PREDICT ERROR
	// ===============================
	action_mask = new placeholder<double>(std::vector<size_t>{n_actions, mini_batch_size}, "action_mask");

	ivariable<double>* inter_mul = new mul<double>(action_scores, *action_mask);
	ivariable<double>* masked_action_score = // reduce sum
		new compress<double>(*inter_mul, 1, [](const std::vector<double>& v) {
			double accum;
			for (double d : v) {
				accum += d;
			}
			return accum;
		});
	ivariable<double>* tempdiff = new sub<double>(*masked_action_score, *future_rewards);
	ivariable<double>* sqrdiff = new mul<double>(*tempdiff, *tempdiff);
	ownership.emplace(inter_mul);
	ownership.emplace(masked_action_score);
	ownership.emplace(tempdiff);
	ownership.emplace(sqrdiff);

	prediction_error = new compress<double>(*sqrdiff); // reduce mean
	// minimize error
	gradient<double>* grads = new gradient<double>(*prediction_error);

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
	expose<double> out(*predicted_actions);
	return out.get_raw();
}

void dq_net::store (
	std::vector<double> observation,
	size_t action_idx,
	double reward,
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

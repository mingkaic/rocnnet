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

void dq_net::variable_setup (nnet::OPTIMIZER<double> optimizer) {
	// ===============================
	// ACTION AND TRAINING VARIABLES!
	// ===============================

	// ACTION SCORE COMPUTATION
	// ===============================
	nnet::ivariable<double>* action_scores = (*target_net)(observation);
	predicted_actions = new compress<double>(action_scores, 1,
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
	action_expose = new expose<double>(predicted_actions);

	// PREDICT FUTURE REWARDS
	// ===============================
	nnet::ivariable<double>* next_action_scores = (*target_net)(next_observation);
	nnet::ivariable<double>* target_values = // reduce max
			new compress<double>(next_action_scores, 1,
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
	nnet::ivariable<double>* future_rewards = PLACEHOLDER_TO_VAR<double>(rewards) + (discount_rate * target_values);

	// PREDICT ERROR
	// ===============================
	nnet::ivariable<double>* masked_action_score = // reduce sum
			new compress<double>(action_scores * PLACEHOLDER_TO_VAR<double>(action_mask), 1,
												[](const std::vector<double>& v) {
													double accum;
													for (double d : v) {
														accum += d;
													}
													return accum;
												});
	nnet::ivariable<double>* diff = masked_action_score - future_rewards;
	prediction_error = new compress<double>(diff * diff); // reduce mean
	// minimize error
	optimizer->ignore(next_action_scores);
	GRAD_MAP<double> gradients = optimizer->compute_grad(prediction_error);

	// clip the gradients to reduce outliers
	for (auto it = gradients.begin(); gradients.end() != it; it++) {
		nnet::ivariable<double>* var = (*it).first;
		nnet::ivariable<double>* grad = (*it).second;
		if (nullptr != grad) {
			(*it).second = new clip_by_norm<double>(grad, 5);
		}
	}

	train_op = optimizer->apply_grad(gradients);

	// UPDATE TARGET NETWORK
	// ===============================
	std::vector<WB_PAIR> q_net_var = q_net->get_variables();
	std::vector<WB_PAIR> target_q_net_var = q_net->get_variables();
	for (size_t i = 0; i < q_net_var.size(); i++) {
		nnet::ivariable<double>* dwt = update_rate * (q_net_var[i].first - target_q_net_var[i].first);
		nnet::ivariable<double>* dbt = update_rate * (q_net_var[i].second - target_q_net_var[i].second);

		WB_PAIR wb = target_q_net_var[i];
		ievoker<double>* w_evok = std::make_shared<update_sub<double> >(
			std::static_pointer_cast<variable<double>, ivariable<double> >(wb.first), dwt);
		ievoker<double>* b_evok = std::make_shared<update_sub<double> >(
			std::static_pointer_cast<variable<double>, ivariable<double> >(wb.second), dbt);
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
	nnet::OPTIMIZER<double> optimizer,
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
	observation = new placeholder<double>(in_shape, "observation");
	next_observation = new placeholder<double>(in_shape, "next_observation");
	// mask and reward shape depends on batch size
	next_observation_mask = new placeholder<double>(std::vector<size_t>{n_observations, 0}, "new_observation_mask");
	rewards = new placeholder<double>(std::vector<size_t>{0}, "rewards");
	action_mask = new placeholder<double>(std::vector<size_t>{n_actions, 0}, "action_mask");

	variable_setup(optimizer);

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

	*observation = observations;
	return action_expose->get_raw();
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
		train_op->eval();

		// update q nets
		net_train.eval();

		iteration++;
	}
	n_train_called++;
}

}

#endif

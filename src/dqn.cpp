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

// def __init__(self, in_size, n_layers, layers_f,
// 		tf_session, optimizer,
// 		train_interval = 5,
// 		discount_rate = 0.95,
// 		model_update_rate = 0.01):
// 	# remember parameters
// 	self.in_size = in_size
// 	self.out_size = n_layers[-1]

// 	# create model
// 	self.q_network = MLayerPerceptron(in_size, n_layers, layers_f)
// 	self.tf_session = tf_session
// 	self.optimizer = optimizer

// 	with tf.name_scope("taking_action"):
// 		self.in_gates		= tf.placeholder(tf.float32, (None, self.in_size), name="in_weight")
// 		self.action_scores	= tf.identity(self.q_network(self.in_gates), name="action_scores")
// 		tf.histogram_summary("action_scores", self.action_scores)
// 		self.predicted_actions  = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")

// 	# training constants
// 	self.train_interval		= train_interval
// 	self.discount_rate		= tf.constant(discount_rate)
// 	self.t_net_update_rate 	= tf.constant(model_update_rate)

// 	# deepq state
// 	self.train_count 		= 0
// 	self.target_q_network	= self.q_network.copy(scope="target_network")

// 	# FOR PREDICTING TARGET FUTURE REWARDS
// 	with tf.name_scope("estimating_future_rewards"):
// 		self.next_in_weight 			= tf.placeholder(tf.float32, (None, self.in_size), name="next_in_weight")
// 		self.next_in_weight_mask		= tf.placeholder(tf.float32, (None,), name="next_in_weight_mask")
// 		self.next_action_scores			= tf.stop_gradient(self.target_q_network(self.next_in_weight))
// 		tf.histogram_summary("target_action_scores", self.next_action_scores)
// 		self.rewards					= tf.placeholder(tf.float32, (None,), name="rewards")
// 		target_values				 	= tf.reduce_max(self.next_action_scores, reduction_indices=[1,]) * self.next_in_weight_mask
// 		self.future_rewards				= self.rewards + self.discount_rate * target_values

// 	# FOR ERROR PREDICTION
// 	with tf.name_scope("q_value_predicition"):
// 		self.action_mask				= tf.placeholder(tf.float32, (None, self.out_size), name="action_mask")
// 		self.masked_action_scores		= tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,])
// 		temp_diff						= self.masked_action_scores - self.future_rewards
// 		self.prediction_error			= tf.reduce_mean(tf.square(temp_diff))

// 		#self.train_op = optimizer.minimize(self.prediction_error)
// 		gradients						= self.optimizer.compute_gradients(self.prediction_error)
// 		for i, (grad, var) in enumerate(gradients):
// 			if grad is not None:
// 				gradients[i] = (tf.clip_by_norm(grad, 5), var)
// 		# Add histograms for gradients.
// 		for grad, var in gradients:
// 			tf.histogram_summary(var.name, var)
// 			if grad is not None:
// 				tf.histogram_summary(var.name + '/gradients', grad)
// 		self.train_op					= self.optimizer.apply_gradients(gradients)

// 	# UPDATE TARGET NETWORK
// 	with tf.name_scope("target_network_update"):
// 		self.target_network_update = []
// 		for v_source, v_target in zip(self.q_network.get_variables(), self.target_q_network.get_variables()):
// 			# this is equivalent to target = (1-alpha) * target + alpha * source
// 			update_op = v_target.assign_sub(self.t_net_update_rate * (v_target - v_source))
// 			self.target_network_update.append(update_op)
// 		self.target_network_update = tf.group(*self.target_network_update)

// 	# summaries
// 	tf.scalar_summary("prediction_error", self.prediction_error)

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
	n_observations(n_input),
	rand_action_prob(rand_action_prob),
	store_interval(store_interval),
	train_interval(train_interval),
	mini_batch_size(mini_batch_size),
	discount_rate(discount_rate),
	max_exp(max_exp),
	update_rate(update_rate) {

	session& sess = session::get_instance();

	IN_PAIR lastpair = *(hiddens.rbegin());
	n_actions = lastpair.first;

	q_net = new ml_perceptron(n_input, hiddens, "q_network");

	actions_executed = 0;
	iteration = 0;
	n_store_called = 0;
	n_train_called = 0;

	// ===============================
	// ACTION AND TRAINING VARIABLES!
	// ===============================
	tensor_shape in_shape = std::vector<size_t>{n_input};
	target_net = q_net->clone("target_network");

	// ACTION SCORE COMPUTATION
	// ===============================
	placeholder<double>* observation =
		new placeholder<double>(in_shape);
	ivariable<double>& action_scores = (*target_net)(*observation);
	ivariable<double>* predicted_actions = // max arg index
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
	placeholder<double>* next_observation = new placeholder<double>(in_shape);
	ivariable<double>& next_action_scores = (*target_net)(*next_observation);
	// unknown shapes
	// mask and reward shape depends on batch size
	placeholder<double>* next_observation_mask =
		new placeholder<double>(std::vector<size_t>{n_observations, mini_batch_size});
	placeholder<double>* rewards =
		new placeholder<double>(std::vector<size_t>{mini_batch_size});
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

	// PREDICT ERROR
	// ===============================
	placeholder<double>* action_mask = new placeholder<double>(std::vector<size_t>(0, n_actions));
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
	ivariable<double>* prediction_error = new compress<double>(*sqrdiff); // reduce mean
	// minimize error

//	gradients                       = self.optimizer.compute_gradients(self.prediction_error)
//	for i, (grad, var) in enumerate(gradients):
//		if grad is not None:
//			gradients[i] = (tf.clip_by_norm(grad, 5), var)
//	self.train_op                   = self.optimizer.apply_gradients(gradients)

	sess.initialize_all<double>();
}

// def __call__(self, xvec):
// 	res = self.tf_session.run(self.predicted_actions, {self.in_gates: xvec[np.newaxis,:]})
// 	return res[0]

std::vector<double> dq_net::operator () (std::vector<double>& input) {
	return std::vector<double>();
}

// def act(self, xvec):
// 	outs = self.tf_session.run(self.action_scores, {self.in_gates: xvec[np.newaxis,:]})
// 	return outs

// def train(self, minibatch, epoch):
// 	if self.train_count % self.train_interval == 0:
// 		states, newstates, newstates_mask, actions, rewards = minibatch
// 		# update internal
// 		cost, _ = self.tf_session.run([
// 			self.prediction_error,
// 			self.train_op,
// 		], {
// 			self.in_gates:				states,
// 			self.next_in_weight:		newstates,
// 			self.next_in_weight_mask:	newstates_mask,
// 			self.action_mask:			actions,
// 			self.rewards:				rewards,
// 		})

// 		print ("epoch=%d, cost = %f" %(epoch, cost))

// 		self.tf_session.run(self.target_network_update)

// 	self.train_count += 1

void dq_net::train (std::vector<std::vector<double> > train_batch) {}

}

#endif

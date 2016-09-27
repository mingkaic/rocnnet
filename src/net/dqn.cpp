//
//  dqn.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/nnet.hpp"

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

dq_net::dq_net (size_t n_input,
				std::vector<std::pair<size_t, ioperation<double> > > hiddens,
				size_t train_interval,
				double discount_rate,
				double update_rate) :
		train_interval(train_interval),
		discount_rate(discount_rate),
		update_rate(update_rate) {
	// create model
	//q_net = new ml_perceptron(n_input, hiddens);

	// input injection

	//
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

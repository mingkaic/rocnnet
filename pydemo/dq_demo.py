#!/usr/bin/env python

import numpy as np

import _init_paths
from tf_rl.controller import DiscreteDeepQ, NL
from collections import deque

def batch_generate(n):
	batch = np.random.random(n)
	return batch

def avgevry2(batch):
	out = np.zeros(len(batch)/2)
	for i, b in enumerate(batch):
		out[i/2] = out[i/2] + b
	return out/2

def data_prep(n):
	X = batch_generate(n)
	Y = avgevry2(X)
	return X, Y

# calculates the circumference distance between A and B assuming A and B represent positions on a circle with circumference wrap_size
def wrapdist(A, B, wrapsize):
	within_dist = max(A - B, B - A)
	wraparound_dist = min(A + wrapsize - B, B + wrapsize - A)
	return min(within_dist, wraparound_dist)

episode_count = 500
max_steps = 100

n_observations = 10
n_actions = 9
n_hiddens = [9, n_actions]
n_acts = [NL.TANH, NL.IDENTITY]
learning_rate = 0.1
decay = 0.9
store_interval = 1
discount_rate = 0.99
exploration_period = 0

current_controller = DiscreteDeepQ(n_observations, n_hiddens, n_acts,
	learning_rate = learning_rate, decay=decay, exploration_period=exploration_period,
	store_every_nth=store_interval, discount_rate=discount_rate)
untrained_controller = DiscreteDeepQ(n_observations, n_hiddens, n_acts,
	learning_rate = learning_rate, decay=decay, exploration_period=exploration_period,
	store_every_nth=store_interval, discount_rate=discount_rate)

current_controller.initialize()
untrained_controller.initialize()

error_queue_size = 10
action_dist = n_actions / 2
error_queue = deque()
for i in range(episode_count):
	avgreward = 0
	observations, expect_out = data_prep(n_observations)

	episode_err = 0
	for j in range(max_steps):
		action = current_controller.action(observations)
		# perform action
		expect_action = np.argmax(expect_out)
		err = wrapdist(expect_action, action, n_actions)
		reward = 1 - 2.0 * err / action_dist
		avgreward = avgreward + reward

		new_observations, expect_out = data_prep(n_observations)

		current_controller.store(observations, action, reward, new_observations)
		current_controller.training_step()

		observations = new_observations
		episode_err = episode_err + float(err)/action_dist

	avgreward = avgreward / max_steps
	episode_err = episode_err / max_steps
	error_queue.append(episode_err)
	if (len(error_queue) > error_queue_size):
		error_queue.popleft()
		# allow ~15% decrease in accuracy (15% increase in error) since last episode
		# otherwise declare that we overfitted and quit
		avgerr = 0
		for last_err in error_queue:
			avgerr = avgerr + last_err
		avgerr = avgerr / len(error_queue)

		if (avgerr - episode_err > 0.15):
			break

	print("episode {} performance: {}% average error, reward: {}".format(i, episode_err * 100, avgreward))

trained_episode_err = 0
untrained_episode_err = 0
for j in range(max_steps):
	observations, expect_out = data_prep(n_observations)
	trained_action = current_controller.action(observations)
	untrained_action = untrained_controller.action(observations)
	# perform action
	expect_action = np.argmax(expect_out)
	trained_err = wrapdist(expect_action, trained_action, n_actions)
	untrained_err = wrapdist(expect_action, untrained_action, n_actions)

	trained_episode_err = trained_episode_err + float(trained_err)/action_dist
	untrained_episode_err = untrained_episode_err + float(untrained_err)/action_dist

print("{}% trained error vs {}% untrained error".format(
	trained_episode_err,
	untrained_episode_err))

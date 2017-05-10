#!/usr/bin/env python

import _init_paths
import gym
from dqn import *

serializedname = 'dqntest.pbx'

spec = gym.spec('CartPole-v0')
env = spec.make()
episode_count = 250
max_steps = 10000

action_space = env.action_space
maxaction = action_space.n

observation_space = env.observation_space
maxobservation = observation_space.shape[0]

hiddens = [5, 5, maxaction]
batchsize = 12 # store at least 12 times before training
controller = dqn_agent(maxobservation, hiddens, 0.9, batchsize, 'pybrain')
controller.initialize(serializedname)

# training step
for ep in xrange(episode_count):
    observation = env.reset()
    reward = done = None
    total_reward = 0
    nsteps = 0
    for step_it in range(max_steps):
        action = controller.action(observation)
        new_observation, reward, done, _ = env.step(action)
        
        controller.store(observation, action, reward, new_observation)
        controller.train()
        
        observation = new_observation
        total_reward = total_reward + reward
        # env.render()
        nsteps = step_it # record step iteration since episode can end early
        if done:
            break
    print 'episode {}: total reward of {} in {} steps'.format(ep, total_reward, nsteps+1)

# controller.save(serializedname)

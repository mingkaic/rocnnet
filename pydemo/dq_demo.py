#!/usr/bin/env python

import _init_paths
import gym
from tf_rl.controller import DiscreteDeepQ, NL

specname = 'CartPole-v0'
serializedname = 'dqntest_'+specname+'.pbx'

spec = gym.spec(specname)
env = spec.make()
episode_count = 250
max_steps = 10000

action_space = env.action_space
maxaction = action_space.n

observation_space = env.observation_space
maxobservation = observation_space.shape[0]

batchsize = 12 # store at least 12 times before training
controller = DiscreteDeepQ(maxobservation, [5, 5, maxaction],
    [NL.SIGMOID, NL.SIGMOID, NL.IDENTITY], learning_rate=0.001, decay=0.9,
    minibatch_size=batchsize, discount_rate=0.99, exploration_period=5000,
    max_experience=10000, store_every_nth=4, train_every_nth=4)
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
        controller.training_step()
        
        observation = new_observation
        total_reward = total_reward + reward
        # env.render()
        nsteps = step_it # record step iteration since episode can end early
        if done:
            break
    print 'episode {}: total reward of {} in {} steps'.format(ep, total_reward, nsteps+1)

# controller.save(serializedname)

import _init_paths
from dqn import *

class NL:
    SIGMOID=0
    TANH=1
    IDENTITY=2

# provide ducky interface for running on tf_rl simulations
class DiscreteDeepQ(object):
    def __init__(self, 
        input_sizes,
        hiddens, 
        nonlinearities,
        learning_rate=0.01,
        decay=0.5,
        random_action_probability=0.05,
        exploration_period=1000,
        store_every_nth=5,
        train_every_nth=5,
        minibatch_size=32,
        discount_rate=0.95,
        max_experience=30000,
        target_network_update_rate=0.01):

        self.agent = dqn_agent(input_sizes, hiddens, nonlinearities, learning_rate,
            decay, random_action_probability, exploration_period, store_every_nth,
            train_every_nth, minibatch_size, discount_rate, max_experience, target_network_update_rate)


    def action(self, observation):
        return self.agent.action(observation)


    def store(self, observation, action, reward, newobservation):
        self.agent.store(observation, action, reward, newobservation)


    def training_step(self):
        self.agent.train()


    def initialize(self, save_dir=""):
        self.agent.initialize(save_dir)


    def save(self, save_dir):
        self.agent.save(save_dir)

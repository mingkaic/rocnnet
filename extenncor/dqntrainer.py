import math
import random
import numpy as np

import tenncor as tc

_get_random = tc.unif_gen(0, 1)

# generalize as feedback class
class DQNEnv:
    def __init__(self, model, sess,
        update_fn, gradprocess_fn = None,
        train_interval = 5, action_prob = 0.05,
        discount_rate = 0.95, target_update_rate = 0.01,
        explore_period = 1000, store_interval = 5,
        mbatch_size = 32, max_exp = 30000):

        self.sess = sess
        self.source_model = model
        self.target_model = model.deep_clone()
        self.train_interval = train_interval
        self.action_prob = action_prob
        self.discount_rate = discount_rate
        self.target_update_rate = target_update_rate
        self.explore_period = explore_period
        self.store_interval = store_interval
        self.mbatch_size = mbatch_size
        self.max_exp = max_exp
        self.actions_executed = 0
        self.ntrain_called = 0
        self.nstore_called = 0
        self.experiences = []

        indim = model.get_input().shape()[-1]
        outdim = model.shape()[-1]
        batchin = [mbatch_size, indim]
        batchout = [mbatch_size, outdim]
        outshape = [mbatch_size]

        self.obs_input = tc.EVariable([indim], 0, 'obs')
        self.src_input = tc.EVariable(batchin, 0, 'src_obs')
        self.nxt_input = tc.EVariable(batchin, 0, 'nxt_obs')
        self.src_outmask = tc.EVariable(batchout, 1, 'src_outmask')
        self.nxt_outmask = tc.EVariable(outshape, 1, 'nxt_outmask')
        self.reward_input = tc.EVariable(outshape, 0, 'rewards')

        # environment interaction
        self.action_idx = tc.argmax(self.source_model.connect(self.obs_input))

        # training
        # forward action score computation
        self.src_act = self.source_model.connect(self.src_input)

        # predicting target future rewards
        self.nxt_act = self.target_model.connect(self.nxt_input)
        target_vals = self.nxt_outmask * tc.reduce_max_1d(self.nxt_act, 0)

        future_reward = self.reward_input + discount_rate * target_vals

        masked_output_score = tc.reduce_sum_1d(self.src_act * self.src_outmask, 0)
        prediction_err = tc.reduce_mean(tc.square(masked_output_score - future_reward))

        source_vars = self.source_model.get_storage()
        target_vars = self.target_model.get_storage()
        assert(len(source_vars) == len(target_vars))
        for sv, tv in zip(source_vars, target_vars):
            assert(hash(sv) != hash(tv))

        source_errs = dict()
        for i, source_var in enumerate(source_vars):
            error = tc.derive(prediction_err, source_var)
            if gradprocess_fn is not None:
                error = gradprocess_fn(error)
            source_errs[source_var] = error

        src_updates = dict(update_fn(source_errs))

        self.updates = []
        for i, (target_var, source_var) in enumerate(zip(target_vars, source_vars)):
            diff = target_var - src_updates[source_var]
            assign = tc.assign_sub(target_var, target_update_rate * diff)
            self.updates.append(assign)

        self.sess.track([prediction_err, self.src_act, self.action_idx] + self.updates)

    def action(self, obs):
        self.actions_executed += 1
        exploration = self._linear_annealing(1.)
        # perform random exploration action
        if _get_random() < exploration:
            return math.floor(_get_random() * self.source_model.shape()[-1])

        self.obs_input.assign(obs)
        self.sess.update()
        return int(self.action_idx.get())

    def store(self, observation, action_idx, reward, new_obs):
        if 0 == self.nstore_called % self.store_interval:
            self.experiences.append((observation, action_idx, reward, new_obs))
            if len(self.experiences) > self.max_exp:
                self.experiences = self.experiences[1:]
        self.nstore_called += 1

    def train(self):
        if len(self.experiences) < self.mbatch_size:
            return
        # extract mini_batch from buffer and backpropagate
        if 0 == (self.ntrain_called % self.train_interval):
            samples = self._random_sample()

            # batch data process
            states = [] # <ninput, batchsize>
            new_states = [] # <ninput, batchsize>
            action_mask = [] # <noutput, batchsize>
            rewards = [] # <batchsize>

            for observation, action_idx, reward, new_obs in samples:
                assert(len(new_obs) > 0)
                states.append(observation)
                local_act_mask = [0.] * self.source_model.shape()[-1]
                local_act_mask[action_idx] = 1.
                action_mask.append(local_act_mask)
                rewards.append(reward)
                new_states.append(new_obs)

            # enter processed batch data
            states = np.array(states)
            new_states = np.array(new_states)
            action_mask = np.array(action_mask)
            rewards = np.array(rewards)
            self.src_input.assign(states)
            self.src_outmask.assign(action_mask)
            self.nxt_input.assign(new_states)
            self.reward_input.assign(rewards)

            self.sess.update_target(self.updates)
        self.ntrain_called += 1

    def _linear_annealing(self, initial_prob):
        if self.actions_executed >= self.explore_period:
            return self.action_prob
        return initial_prob - self.actions_executed * (initial_prob - self.action_prob) / self.explore_period

    def _random_sample(self):
        return random.sample(self.experiences, self.mbatch_size)

import math
import random
import numpy as np

import extenncor.trainer_cache as ecache
import extenncor.dqntrainer_pb2 as dqn_pb

import tenncor as tc

_get_random = tc.unif_gen(0, 1)

# generalize as feedback class
@ecache.EnvManager.register
class DQNEnv(ecache.EnvManager):
    def __init__(self, src_model, sess,
        update_fn, gradprocess_fn = None,
        optimize_cfg = "", max_exp = 30000,
        train_interval = 5, store_interval = 5,
        explore_period = 1000, action_prob = 0.05,
        mbatch_size = 32, discount_rate = 0.95,
        target_update_rate = 0.01, clean_startup = False,
        cachedir = '/tmp'):

        self.max_exp = max_exp
        self.train_interval = train_interval
        self.store_interval = store_interval
        self.explore_period = explore_period
        self.action_prob = action_prob

        def default_init():
            self.actions_executed = 0
            self.ntrain_called = 0
            self.nstore_called = 0
            self.experiences = []

            nxt_model = src_model.deep_clone()

            inshape = list(src_model.get_input().shape())
            batchin = [mbatch_size] + inshape

            # environment interaction
            self.obs = tc.EVariable(inshape, 0, 'obs')
            self.act_idx = tc.argmax(src_model.connect(self.obs))
            self.act_idx.tag("recovery", "act_idx")

            # training
            self.src_obs = tc.EVariable(batchin, 0, 'src_obs')
            self.nxt_obs = tc.EVariable(batchin, 0, 'nxt_obs')
            self.src_outmask = tc.EVariable([mbatch_size] + list(src_model.shape()), 1, 'src_outmask')
            self.nxt_outmask = tc.EVariable([mbatch_size], 1, 'nxt_outmask')
            self.rewards = tc.EVariable([mbatch_size], 0, 'rewards')

            # forward action score computation
            src_act = src_model.connect(self.src_obs)

            # predicting target future rewards
            nxt_act = nxt_model.connect(self.nxt_obs)
            target_vals = self.nxt_outmask * tc.reduce_max_1d(nxt_act, 0)

            future_reward = self.rewards + discount_rate * target_vals

            masked_output_score = tc.reduce_sum_1d(src_act * self.src_outmask, 0)
            prediction_err = tc.reduce_mean(tc.square(masked_output_score - future_reward))

            source_vars = src_model.get_storage()
            target_vars = nxt_model.get_storage()
            assert(len(source_vars) == len(target_vars))

            source_errs = dict()
            for source_var in source_vars:
                error = tc.derive(prediction_err, source_var)
                if gradprocess_fn is not None:
                    error = gradprocess_fn(error)
                source_errs[source_var] = error

            src_updates = dict(update_fn(source_errs))

            updates = [prediction_err, src_act, self.act_idx]
            for target_var, source_var in zip(target_vars, source_vars):
                diff = target_var - src_updates[source_var]
                assign = tc.assign_sub(target_var, target_update_rate * diff)
                updates.append(assign)

            self.sess.track(updates)
            tc.optimize(self.sess, optimize_cfg)

        super().__init__('dqn', sess, default_init=default_init,
            clean=clean_startup,  cacheroot=cachedir)
        self.src_shape = self.src_outmask.shape()
        self.mbatch_size = self.rewards.shape()[0]

    def _backup_env(self, fpath: str) -> bool:
        with open(fpath, 'wb') as envfile:
            dqn_env = dqn_pb.DqnEnv()

            dqn_env.actions_executed = self.actions_executed
            dqn_env.ntrain_called = self.ntrain_called
            dqn_env.nstore_called = self.nstore_called
            for obs, act_idx, reward, new_obs in self.experiences:
                exp = dqn_env.experiences.add()
                for ob in obs:
                    exp.obs.append(ob)
                for nob in new_obs:
                    exp.new_obs.append(nob)
                exp.act_idx = act_idx
                exp.reward = reward

            envfile.write(dqn_env.SerializeToString())
            return True
        return False

    def _recover_env(self, fpath: str) -> bool:
        # recover object members from recovered session
        query = tc.Statement(self.sess.get_tracked())
        self.obs = query.find('{ "leaf":{ "label":"obs" } }')[0]
        self.act_idx = query.find('{ "op":{ "opname":"ARGMAX", "attrs":{"recovery":"act_idx" } }')[0]
        self.src_obs = query.find('{ "leaf":{ "label":"src_obs" } }')[0]
        self.nxt_obs = query.find('{ "leaf":{ "label":"nxt_obs" } }')[0]
        self.src_outmask = query.find('{ "leaf":{ "label":"src_outmask" } }')[0]
        self.nxt_outmask = query.find('{ "leaf":{ "label":"nxt_outmask" } }')[0]
        self.rewards = query.find('{ "leaf":{ "label":"rewards" } }')[0]

        with open(fpath, 'rb') as envfile:
            dqn_env = dqn_pb.DqnEnv()
            dqn_env.ParseFromString(envfile.read())

            self.actions_executed = dqn_env.actions_executed
            self.ntrain_called = dqn_env.ntrain_called
            self.nstore_called = dqn_env.nstore_called
            self.experiences = [(exp.obs, exp.act_idx, exp.reward, exp.new_obs)
                for exp in dqn_env.experiences]
            return True
        return False

    def action(self, obs):
        self.actions_executed += 1
        exploration = self._linear_annealing(1.)
        # perform random exploration action
        if _get_random() < exploration:
            return math.floor(_get_random() * self.src_shape[-1])

        self.obs.assign(obs)
        self.sess.update()
        return int(self.act_idx.get())

    def store(self, observation, act_idx, reward, new_obs):
        if 0 == self.nstore_called % self.store_interval:
            self.experiences.append((observation, act_idx, reward, new_obs))
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

            for observation, act_idx, reward, new_obs in samples:
                assert(len(new_obs) > 0)
                states.append(observation)
                local_act_mask = [0.] * self.src_shape[-1]
                local_act_mask[act_idx] = 1.
                action_mask.append(local_act_mask)
                rewards.append(reward)
                new_states.append(new_obs)

            # enter processed batch data
            states = np.array(states)
            new_states = np.array(new_states)
            action_mask = np.array(action_mask)
            rewards = np.array(rewards)
            self.src_obs.assign(states)
            self.src_outmask.assign(action_mask)
            self.nxt_obs.assign(new_states)
            self.rewards.assign(rewards)

            self.sess.update()
        self.ntrain_called += 1

    def _linear_annealing(self, initial_prob):
        if self.actions_executed >= self.explore_period:
            return self.action_prob
        return initial_prob - self.actions_executed * (initial_prob - self.action_prob) / self.explore_period

    def _random_sample(self):
        return random.sample(self.experiences, self.mbatch_size)

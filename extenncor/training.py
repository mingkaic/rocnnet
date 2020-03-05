import os.path

import numpy as np

import tenncor as tc

cache_dir = '/tmp'

def get_rand_seed(rng):
    seed = 0
    seed_options = dir(rng)
    if 'get_state' in seed_options:
        seed = rng.get_state()
    elif 'random' in seed_options:
        seed = rng.random()
    else:
        try:
            seed = rng()
        except:
            print('warning: no state option in {}'.format(rng))
    pass

def recover_np_seed(rng, state):
    seed_options = dir(rng)
    if 'seed' in seed_options:
        rng.seed(state.seed)
    else:
        try:
            rng(state.seed)
        except:
            print('warning: no seed option in {}'.format(rng))

def cache(session, env):
    model_target = os.path.join(cache_dir, 'session.onnx')
    try:
        print('caching model')
        if tc.save_session_file(model_target, session):
            print('successfully cached to {}'.format(model_target))
    except Exception as e:
        print(e)
        print('failed to cache to "{}"'.format(model_target))


def recover(model, env):
    pass

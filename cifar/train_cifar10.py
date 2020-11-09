# Train cifar10 through dataset
import os

import tenncor as tc

from extenncor.dataset_trainer import OnxDSEnv
import onnxds.read_dataset as helper

import numpy as np
import matplotlib.pyplot as plt

from distr_instances.cnn_init import DistrCNNModel as CNNModel

env_offset = os.getenv('CNN_TRAIN_OFFSET', 0)

train_oxfile = 'models/cifar_train.onnx'
test_oxfile = 'models/cifar_test.onnx'
nbatch = 5
learning_rate = 0.01
l2_decay = 0.0001
show_every_n = 5
nepochs = 10
backup_every = 50
test_every = 10
train_fepoch_errs = []
train_lepoch_errs = []
train_avg_errs = []
last_err = None

cnn = CNNModel()

test_ds = helper.load(test_oxfile)
test_iteration = cnn.make_test_iterator(test_ds)

def _cifar_trainstep(train_idx, ctx, data, trainins, trainouts):
    if train_idx % backup_every == backup_every - 1:
        env.backup()

    global last_err
    invar, exout = tuple(trainins)
    trainerr = trainouts[0]

    labels = np.zeros((nbatch, 10))
    for j, label in enumerate(data['label']):
        labels[j][label] = 1
    invar.assign(data['image'].astype(np.float))
    exout.assign(labels.astype(np.float))
    epoch_errs = []
    last_epocherr = None
    for j in range(nepochs):
        err = trainerr.get()

        print('==== epoch {} ===='.format(j))
        print('error: {}'.format(err))

        # compare with historic data to ensure training stability
        if last_err is not None and np.array_equal(err, last_err):
            print('amazing coincedence!')

        if last_epocherr is not None and np.any(last_epocherr < err):
            print('last epoch for the same sample had better performance')

        last_err = err
        last_epocherr = err
        epoch_errs.append(np.average(err))

    first_err = epoch_errs[0]
    last_err = epoch_errs[-1]
    avg_err = np.average(epoch_errs)
    train_fepoch_errs.append(first_err)
    train_lepoch_errs.append(last_err)
    train_avg_errs.append(avg_err)
    if train_idx % show_every_n == show_every_n - 1:
        print('==== {}th image ====\nfirst epoch err:{}\nlast epoch error:{}\naverage error:{}'.format(
            train_idx, first_err, last_err, avg_err))

    if train_idx % test_every == test_every - 1:
        test_iteration()

raw_inshape = cnn.model_input.shape()
trainin = tc.EVariable([nbatch] + raw_inshape, label='trainin')
trainexout = tc.EVariable([nbatch, 10], label='trainexout')
env = OnxDSEnv('cifar10', train_oxfile, [trainin, trainexout],
            cnn.make_error_setup(learning_rate, l2_decay),
            _cifar_trainstep,
            optimize_cfg='cfg/optimizations.json')
for _ in range(env_offset):
    next(env.dataset)

try:
    while env.train():
        pass
except KeyboardInterrupt:
    print("Interrupted")

plt.plot(list(range(len(train_fepoch_errs))), train_fepoch_errs)
plt.plot(list(range(len(train_lepoch_errs))), train_lepoch_errs, 'bo')
plt.plot(list(range(len(train_avg_errs))), train_avg_errs, 'r+')
test_iteration(show=True)

try:
    print('saving')
    target = 'models/cifar10.onnx'
    if tc.save_to_file(target, [cnn.model]):
        print('successfully saved to {}'.format(target))
except Exception as e:
    print(e)
    print('failed to write to "{}"'.format(target))

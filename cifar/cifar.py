import tenncor as tc

import onnxds.read_dataset as helper

import logging
import numpy as np
import matplotlib.pyplot as plt
from aliaser import AliasService
from control import ControlServer

names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

def cross_entropy_loss(Label, Pred):
    return -tc.api.reduce_sum(Label * tc.api.log(Pred + np.finfo(float).eps), set([0]))

class PersistentCifarModel:
    def __init__(self, consul_port, aliaser, inshape):
        self.mgr = tc.DistrManager(tc.Consul(), port=consul_port,
            service_name='cifar_cnn', alias='master')
        tc.set_distrmgr(self.mgr)
        # batch, height, width, in
        # construct CNN
        paddings = ((2, 2), (2, 2))
        self.model_input = tc.EVariable(inshape, label='model_input')
        self.model = tc.api.layer.link([ # minimum input shape of [1, 32, 32, 3]
            tc.api.layer.bind(lambda x: x / 255. - 0.5), # normalization
            tc.api.layer.conv([5, 5], 3, 16,
                weight_init=tc.api.layer.norm_xavier_init(0.5),
                zero_padding=paddings), # outputs [nbatch, 32, 32, 16]
            tc.api.layer.bind(tc.api.relu),
            tc.api.layer.bind(lambda x: tc.api.nn.max_pool2d(x, [1, 2]),
                inshape=tc.Shape([1, 32, 32, 16])), # outputs [nbatch, 16, 16, 16]
            tc.api.layer.conv([5, 5], 16, 20,
                weight_init=tc.api.layer.norm_xavier_init(0.3),
                zero_padding=paddings), # outputs [nbatch, 16, 16, 20]
            tc.api.layer.bind(tc.api.relu),
            tc.api.layer.bind(lambda x: tc.api.nn.max_pool2d(x, [1, 2]),
                inshape=tc.Shape([1, 16, 16, 20])), # outputs [nbatch, 8, 8, 20]
            tc.api.layer.conv([5, 5], 20, 20,
                weight_init=tc.api.layer.norm_xavier_init(0.1),
                zero_padding=paddings), # outputs [nbatch, 8, 8, 20]
            tc.api.layer.bind(tc.api.relu),
            tc.api.layer.bind(lambda x: tc.api.nn.max_pool2d(x, [1, 2]),
                inshape=tc.Shape([1, 8, 8, 20])), # outputs [nbatch, 4, 4, 20]

            tc.api.layer.dense([4, 4, 20], [10], # weight has shape [10, 4, 4, 20]
                weight_init=tc.api.layer.norm_xavier_init(0.5),
                dims=[[0, 1], [1, 2], [2, 3]]), # outputs [nbatch, 10]
            tc.api.layer.bind(lambda x: tc.api.softmax(x, 0, 1))
        ], self.model_input)
        aliaser.alias('cnn_model_input', tc.expose_node(self.model_input))
        aliaser.alias('cnn_model', tc.expose_node(self.model))

    def save(self, filepath):
        return tc.save_to_file(filepath, [self.model])

class TrainingCifarModel:
    def __init__(self, consul_port, aliaser, alias=None):
        self.mgr = tc.DistrManager(tc.Consul(), port=consul_port,
            service_name='cifar_cnn', alias=alias)
        tc.set_distrmgr(self.mgr)
        self.model_input = tc.lookup_node(aliaser.dealias('cnn_model_input'))
        self.model = tc.lookup_node(aliaser.dealias('cnn_model'))

    def make_test_iterator(self, ds):
        inshape = self.model_input.shape()
        testin = tc.EVariable(inshape, label='testin')
        testexout = tc.EVariable([10], label='testexout')
        testout = self.model.connect(testin)
        testerr = cross_entropy_loss(testexout, testout)
        testidx = tc.api.argmax(testout)
        def test_iteration(show=False):
            test_sample = next(ds)
            labels = np.zeros(10)
            labels[test_sample['label'][0]] = 1

            testin.assign(test_sample['image'].astype(np.float))
            testexout.assign(labels.astype(np.float))
            print('expect: {}'.format(names[test_sample['label'][0]]))
            print('got: {}'.format(names[int(testidx.get())]))
            print('probability: {}'.format(testout.get()))
            print('err: {}'.format(testerr.get()))

            if show:
                plt.imshow(test_sample['image'].reshape(*inshape))
                plt.show()
        return test_iteration

    def make_error_setup(self, learning_rate, l2_decay):
        def error_connect(input_vars, ctx):
            opt = lambda error, leaves: tc.TenncorAPI(ctx).\
                approx.adadelta(error, leaves, step_rate=learning_rate, decay=l2_decay)
            invar, exout = tuple(input_vars)
            trainout = self.model.connect(invar)
            train_err = tc.apply_update([trainout], opt,
                lambda models: cross_entropy_loss(exout, models[0]),
                ctx=ctx)
            return [train_err]
        return error_connect

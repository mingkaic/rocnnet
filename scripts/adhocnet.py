#!usr/bin/env python

import math
import numpy as np
import time

def sigmoid (x):
	return 1/(1+math.exp(-x))

def dsigmoid (x):
	sx = sigmoid(x)
	return sx * (1-sx)

learning_rate = 0.9
ninput = 10
nhidden = 9
noutput = 5
W0 = 2 * np.random.random((nhidden, ninput)) - 1
B0 = 2 * np.zeros(nhidden) - 1
W1 = 2 * np.random.random((noutput, nhidden)) - 1
B1 = 2 * np.zeros(noutput) - 1

def forward (invec):
	# hidden = <W0, in> + B0
	hidden = np.matmul(W0, invec) + B0
	# activation = sigmoid(hidden)
	act = map(sigmoid, hidden)
	# hidden = <W1, in> + B1
	hidden2 = np.matmul(W1, act) + B1
	# output = sigmoid(hidden2)
	out = map(sigmoid, hidden2)
	return (out, hidden2, act, hidden)

def backward (invec, expectout):
	global learning_rate, ninput, nhidden, noutput, W1, W0, B1, B0
	(out, hidden2, act, hidden) = forward(invec)
	l2_error = expectout - out
	# db1 = 2 * (expectout - out) * sigmoid'(W1(sigmoid(<W0, in>+B0))+B1) * 1
	db1 = 2 * l2_error * map(dsigmoid, hidden2)
	# dw1 = <db1, transpose(sigmoid(<W0, in>+B0))>
	dw1 = np.matmul(np.reshape(db1, (noutput, 1)), np.reshape(act, (1, nhidden)))
	# db0 = sigmoid'(<W0, in>+B0) * <db1, W1>
	db0 = map(sigmoid, hidden) * np.matmul(np.transpose(W1), db1)
	# dw0 = <db0, transpose(in)>
	dw0 = np.matmul(np.reshape(db0, (nhidden, 1)), np.reshape(invec, (1, ninput)))

	db0 = np.reshape(db0, (nhidden))

	# X_n+1 = X_n - learning * d
	B1 += learning_rate * db1
	W1 += learning_rate * dw1
	B0 += learning_rate * db0
	W0 += learning_rate * dw0

def output_protocol (invec):
	global noutput
	outvec = []
	for i in range(noutput):
		a = invec[i*2]
		b = invec[i*2+1]
		outvec.append((a + b) / 2)
	outvec = np.array(outvec, dtype=np.float)
	return outvec

ntrain = 60000
ins = []
outs = []
for _ in range(ntrain):
	inv = np.random.random(ninput)
	outv = output_protocol(inv)
	ins.append(inv)
	outs.append(outv)

start = time.clock()
for i in range(ntrain):
	invec = ins[i]
	outvec = outs[i]
	backward(invec, outvec)
elapse = time.clock() - start
print "training time: {} seconds".format(elapse)

ntests = 500
totalerr = 0
for _ in range(ntests):
	invec = np.random.random(ninput)
	(out, _, _, _) = forward(invec)
	outvec = output_protocol(invec)
	err = abs(outvec - out)
	totalerr += np.average(err)

totalerr *= 100.0 / ntests
print "{}% error".format(totalerr)

import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tenncor as tc
import tenncor as eteq

matrix_dims = [
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    500,
    1000,
    1500,
]

np_durs = []
eteq_durs = []
tf_durs = []
for matrix_dim in matrix_dims:
    shape = [matrix_dim, matrix_dim]
    data = np.random.rand(*shape)
    data2 = np.random.rand(*shape)

    var = eteq.variable(data, 'var')
    var2 = eteq.variable(data2, 'var2')
    tf_var = tf.Variable(data)
    tf_var2 = tf.Variable(data2)

    sess = tf.compat.v1.Session()
    sess.run(tf_var.initializer)
    sess.run(tf_var2.initializer)

    # tenncor matmul setup
    out = tc.api.matmul(var, var2)

    # tensorflow matmul setup
    tf_out = tf.matmul(tf_var, tf_var2)

    # numpy matmul calculate
    start = time.time()
    print(data.dot(data2))
    np_dur = time.time() - start

    # tenncor matmul calculate
    start = time.time()
    print(out.get())
    eteq_dur = time.time() - start

    # tensorflow matmul calculate
    start = time.time()
    tf_fout = sess.run(tf_out)
    print(tf_fout)
    tf_dur = time.time() - start

    np_durs.append(np_dur)
    eteq_durs.append(eteq_dur)
    tf_durs.append(tf_dur)

print('numpy durations: ', np_durs)
print('eteq durations: ', eteq_durs)
print('tf durations: ', tf_durs)
ead_line = plt.plot(matrix_dims, eteq_durs, 'r--', label='eteq durations')
np_lines = plt.plot(matrix_dims, np_durs, 'g--', label='numpy durations')
tf_line = plt.plot(matrix_dims, tf_durs, 'b--', label='tf durations')
plt.legend()
plt.show()

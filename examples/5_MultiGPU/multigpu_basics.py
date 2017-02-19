from __future__ import print_function
'''
Basic Multi GPU computation example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

'''
This tutorial requires your machine to have 2 GPUs
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
"/gpu:1": The second GPU of your machine
'''

import numpy as np
import tensorflow as tf
import datetime

from tensorflow.python.client import device_lib

#========================
# How many GPUs
# Get available GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# print GPUs
print(get_available_gpus())
#========================

# Processing Units logs
log_device_placement = False

# Num of multiplications to perform
n = 3

'''
Example: compute A^n + B^n on 2 GPUs
'''
# Create random large matrix
# size = 10000
size = 1000

def matpow(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

def sumpow(n, size):
    A = np.random.rand(size, size).astype('float32')
    B = np.random.rand(size, size).astype('float32')

    # Create a graph to store results
    c1 = []
    a = tf.placeholder(tf.float32, [size, size])
    b = tf.placeholder(tf.float32, [size, size])
    # Compute A^n and B^n and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))
    sum = tf.add_n(c1)
    return sum, a, b, A, B

#Single GPU computing

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    # sess.run(sum, {a:A, b:B})
    with tf.device('/gpu:0'):
        sum, a, b, A, B = sumpow(n, size)
    sess.run(sum, {a: A, b: B})

t2_1 = datetime.datetime.now()
print("Single GPU computation time: " + str(t2_1-t1_1))


#Single CPU computing

t2_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    with tf.device('/cpu:0'):
        sum1, a1, b1, A1, B1 = sumpow(n, size)
    sess.run(sum1, {a1: A1, b1: B1})

t3_1 = datetime.datetime.now()
print("Single CPU computation time: " + str(t3_1-t2_1))


'''
Multi GPU computing
# GPU:0 computes A^n
'''
'''
A3 = np.random.rand(size, size).astype('float32')
B3 = np.random.rand(size, size).astype('float32')
c3 = []

with tf.device('/gpu:0'):
    # Compute A^n and store result in c2
    a3 = tf.placeholder(tf.float32, [size, size])
    c3.append(matpow(a3, n))

# GPU:1 computes B^n
with tf.device('/gpu:1'):
    # Compute B^n and store result in c2
    b3 = tf.placeholder(tf.float32, [size, size])
    c3.append(matpow(b3, n))

with tf.device('/cpu:0'):
    sum3 = tf.add_n(c3) #Addition of all elements in c2, i.e. A^n + B^n

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum3, {a3:A3, b3:B3})
t2_2 = datetime.datetime.now()

print("Multi GPU computation time: " + str(t2_2-t1_2))
'''

#!/usr/bin/env python3

import random as r
import matplotlib.pyplot as plt
import numpy as np
from functools import partial # to bind functions
import math
from copy import copy 

import mnist
import gradient_method
import quick_Nesterov_method
import gradient_method_strongly_convex
import adadelta
import adam

from support import *
import time

dimX = 28*28 + 1
L = 0.5 # 1/(k)
mu = 0.0001
N_algo = 3000
N_train = 1000

def error_func(x, y, w):
	return np.log(1 + np.exp(-y * np.dot(x, w)))

def error_E(x_s, y_s, w):
	return sum(error_func(x_s[i], y[i], w) for i in range(len(x_s)))

def grad_w_error_func(x, y, w):
	tmp = y * np.dot(x, w)
	if tmp > 100:
		return 0*x

	C_ =  -y / (np.exp(tmp) + 1)
	return C_ * x

def grad_w_error_E(x_s, y_s, w):
	return sum(grad_w_error_func(x_s[i], y_s[i], w) for i in range(len(x_s)))

def save_vec(vec, filename):
	with open(filename, "w") as f:
		for v in vec:
			f.write(str(v) + ' ')
			# f.write()

def get_vecs(file):
	f = open(file, 'r')
	vec = []
	for line in f:
		for val in line.split():
			vec.append(float(val))

	return vec


def brush_labels(labels, num):
	labels_ = [1 if labels[i] == num else -1 for i in range(len(labels))]
	return labels_

def train(method_name):
	w = []
	train_set_matrix = mnist.train_images()
	train_set = [train_set_matrix[i].flatten() for i in range(N_train)]
	train_set = [np.append(train_set_matrix[i].flatten(), 1) for i in range(N_train)]
	# train_set = [train_set[i]/np.max(train_set[i]) for i in range(N_train)]

	label_set = mnist.train_labels()
	label_set_num = []
	# train_set_num = []
	# right_label = [1 for _ in range(N_train)]
	for i in range(10):
		label_set_num.append(brush_labels(label_set, i))
		# train_set_num.append([])
		# for j in range(N_train):
			# if label_set_num[i][j] == 1:
				# train_set_num[-1].append(train_set[j])

	w0 = np.array([0 for _ in range(dimX)])

	for i in range(10):
		print(i)
		# grad_error_E_train = partial(grad_w_error_E, train_set, label_set_num[i])
		grad_error_E_train = lambda w : grad_w_error_E(train_set, label_set_num[i], w) + mu*w

		if method_name == 'quick_Nesterov_method':
			w.append(quick_Nesterov_method.algo(grad_error_E_train, w0 , N=N_algo, coef=lambda k: .5))
		elif method_name == 'gradient_method':
			w.append(gradient_method.algo(grad_error_E_train, w0 , N=N_algo, coef=lambda k: .5))
		elif method_name == 'gradient_method_strongly_convex':
			strong_convex_grad_error_E_train = lambda w : grad_w_error_E(train_set, label_set_num[i], w) + mu*w
			w.append(gradient_method_strongly_convex.algo(strong_convex_grad_error_E_train, \
					w0 , L, mu, N=N_algo))
		elif method_name == 'adadelta':
			w.append(adadelta.algo(grad_error_E_train, w0 , lambda k: .5, N=N_algo))
		elif method_name == 'adam':
			w.append(adam.algo(grad_error_E_train, w0 , lambda k: .5, N=N_algo))

		save_vec(w[i], './out/' + method_name + '_w_' + str(i))

def play(method_name, takes_time):
	w = []
	for i in range(10):
		w.append(get_vecs('./out/' + method_name + '_w_' + str(i)))
		# print(w[i])

	test_set_matrix = mnist.test_images()
	test_set = [test_set_matrix[i].flatten() for i in range(len(test_set_matrix))]
	test_set = [np.append(test_set_matrix[i].flatten(), 1) for i in range(len(test_set_matrix))]
	# test_set = [test_set[i]/np.max(test_set[i]) for i in range(len(test_set))]
	# for i in range(len(test_set)):
		# test_set[i] = np.insert(test_set[i], 0, 1)

	label_set = mnist.test_labels()
	bien_respond = 0


	path_ = './result/result_' + method_name
	with open(path_, "w") as f:
		f.write('L = %f, train = %i, N_algo = %i\n' % (L, N_train, N_algo))
		f.write('Takes time %f\n' % takes_time)
		for j in range(len(test_set)):
			vals = [np.dot(test_set[j], w[i]) for i in range(10)]
			argmax = np.argmax(vals)
			# f.write(str(argmax) + ' ' + str(label_set[j]) + '\n')
			bien_respond += int(argmax == label_set[j])

		f.write(str(bien_respond) + ' ' + str(len(test_set)) + ' ' + str(bien_respond / len(test_set)))


if __name__ == '__main__':
	method_name = 'quick_Nesterov_method'
	start_time = time.time()  
	train(method_name)
	takes_time = time.time() - start_time
	play(method_name, takes_time)

	method_name = 'gradient_method'
	start_time = time.time()  
	train(method_name)
	takes_time = time.time() - start_time
	play(method_name, takes_time)

	method_name = 'gradient_method_strongly_convex'
	start_time = time.time()  
	train(method_name)
	takes_time = time.time() - start_time
	play(method_name, takes_time)

	method_name = 'adadelta'
	start_time = time.time()  
	train(method_name)
	takes_time = time.time() - start_time
	play(method_name, takes_time)

	method_name = 'adam'
	start_time = time.time()  
	train(method_name)
	takes_time = time.time() - start_time
	play(method_name, takes_time)

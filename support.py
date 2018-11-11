#!/usr/bin/env python3
import numpy as np
from functools import partial # to bind functions
from math import *

"""
We are to follow next conventions:
1. all vector are multiplied from right. A.v = y
2. vector-vector multiplication is always a dot product
"""

def check_vector(vec):
	try:
		return len(vec[0])
	except:
		return 0

def trans(A):
	return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def norm_vector2(v):
	return dot_prod(v, v)

def dot_prod(v, w):
	return sum(v[i]*w[i] for i in range(len(v)))

def prod(A, B):
	nA, nB = len(A), len(B)
	mA, mB = check_vector(A), check_vector(B)

	if mA == mB == 0:
		return dot_prod(A, B)

	if mA != nB:
		raise IndexError('mA != nB in matrix multiplication')
	if mA == 0:
		raise IndexError('mA == 0 in matrix multiplication')

	if mB == 0:
		return prod_vector(A, B)

	Res = [[0 for _ in range(mB)] for _ in range(nA)]

	for i in range(nA):
		for j in range(mB):
			for k in range(mA):
				Res[i][j] += A[i][k] * B[k][j]

	return Res
	
def prod_vector(A, v):
	res = [0 for _ in range(len(A))]
	for i in range(len(A)):
		for j in range(len(v)):
			res[i] += A[i][j]*v[j]

	return res

def norm(A):
	"""only matrix"""
	return max([sum(abs(A[i][j]) for j in range(len(A[0]))) for i in range(len(A))])


def inv_2(A):
	det = A[0][0]*A[1][1] - A[1][0]*A[0][1]
	return [[A[1][1]/det, -A[0][1]/det], 
			[-A[1][0]/det, A[0][0]/det]]

def pseudo_inverse(A):
	return prod(inv(prod(trans(A), A)), trans(A))

def inv(A):
	return np.linalg.inv(A)

def direction_derivative(func, delta):
	len_delta = sqrt(dot_prod(delta, delta))
	return lambda x: (func(x+delta) - func(x-delta))/(2*len_delta)


def grad_1(func):
	delta = 0.0001
	return lambda x: (func(x+delta) - func(x-delta))/(2*delta)

def grad(func, N):
	if N == 1:
		return grad_1(func)
	delta = 0.0001
	directions = [[(0 if i != j else delta) for i in range(N)] for j in range(N)]
	
	direction_derivative_func = partial(direction_derivative, func)
	grads = list(map(direction_derivative_func, directions))
	return lambda x : np.array([grads[i](x) for i in range(N)])


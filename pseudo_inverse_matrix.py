#!/usr/bin/env python3
from support import *
import numpy as np

"""
in a way this algorithm is very monotone
"""
def algo(A, y, N): 
	""" A - matrix, y - vector, N - number of steps """
	x = np.array([1 for _ in range(len(A[0]))])
	normA = norm(A)**2
	trA = trans(A)
	print(normA)
	for k in range(N):
		x = k/(k+1) * (x - prod(trA, prod_vector(A, x) - y)/normA)

	return x


def tester():
	A = np.array([[1,2], [1,2], [4,6]])
	y = np.array([-1, 1, 4])
	
	print(algo(A, y, 10000))
	print(prod_vector(pseudo_inverse(A), y))
	# in case of [1, 2], [1,2]
	# print(prod_vector([[1/10, 1/10], [2/10, 2/10]], y)) 

tester()
#!/usr/bin/env python3

from support import *
from quick_Nesterov_method import algo


A = np.array([[1,4], [1,2]])
y = np.array([-1, 1])

func = lambda x: 1/2 * norm_vector2(prod_vector(A, x) - y)
L = norm(prod(trans(A), A))

x = algo(grad(func, 2), x0 = [1,1], L = L, N = 1000)
print(x, func(x))

x = prod_vector(pseudo_inverse(A), y) 
print(x, func(x))

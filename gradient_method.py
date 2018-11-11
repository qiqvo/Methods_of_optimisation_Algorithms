#!/usr/bin/env python3
from support import *
import numpy as np

def algo(gfunc, x0, coef = lambda k : 1/k**2, N = 100):
	x= np.array(x0)
	for k in range(1,N+1):
		x = x - coef(k) * gfunc(x)

	return x


def tester():
	func1 = lambda x : (x - 2)**2
	L1 = 2
	func2 = lambda x : 1993*(x - 2)**2 + 4*x
	L2 = 2*1993
	func3 = lambda x: (x[0]*x[1] - 2)**2 - 2*x[0]*x[1]
	L3 = 20
	
	x = algo(grad(func1, 1), x0=1, coef=lambda k : 1/k, N=10)
	print(x, func1(x))

	# if coef is different -- we might have some troubles
	x = algo(grad(func2, 1), x0=1, coef=lambda k : 1/L2, N=10)
	print(x, func2(x))
	x = algo(grad(func3, 2), x0=[1,1], coef=lambda k : 1/L3,N= 100)
	print(x, func3(x))

tester()
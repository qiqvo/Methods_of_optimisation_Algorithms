#!/usr/bin/env python3
from support import *
import numpy as np

eps = 0.000001

def algo(gfunc, x0, coef_L, N=100, alpha=0.5):
	x = np.array(x0)
	G = x - x

	for k in range(N):
		grad = gfunc(x)
		if norm_vector2(grad) < 0.0001:
			break
			
		G = alpha*G + (1 - alpha)*grad**2
		temp = np.sqrt(G + eps)
		temp = coef_L(k+1) / temp
		x = x - temp* grad

	return x
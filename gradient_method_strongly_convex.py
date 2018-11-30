#!/usr/bin/env python3
from support import *
import numpy as np

def algo(gfunc, x0, coef_L, coef_m, N=100):
	x = np.array(x0)
	coef = coef_L + coef_m
	coef = 2/coef

	for k in range(N):
		grad = gfunc(x)
		if norm_vector2(grad) < 0.0001:
			break
			
		x = x - coef * grad

	return x
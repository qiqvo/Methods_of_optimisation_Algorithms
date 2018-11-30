#!/usr/bin/env python3
from support import *
import numpy as np

eps = 0.000001

def algo(gfunc, x0, coef_L, N=100, alpha=0.5, N_av_forget=200):
	x = np.array(x0)
	G = x - x
	av_n = 1
	av_grad = x - x
	av_grad_tail = x - x

	for k in range(N):
		grad = gfunc(x)
		if norm_vector2(grad) < 0.0001:
			break

		av_grad = av_n/(av_n + 1)*av_grad + grad/(av_n+1)
		# av_n += 1
		# if av_n > N_av_forget:
			# av_grad = av_grad - 1/av_n * av_grad_tail
			# av_grad_tail = av_grad
			# av_n -= 1
			
		G = alpha*G + (1 - alpha)*grad**2
		temp = np.sqrt(G + eps)
		temp = coef_L(k+1) / temp
		x = x - temp* av_grad

	return x
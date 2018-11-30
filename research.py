#!/usr/bin/env python3

import matplotlib.pyplot as plt

lam = 0.1
N = 1000

x = [1]
y = [1]

x_ = []
y_ = []

for _ in range(N):
	# поправка y_k
	x_.append(x[-1] - lam * y[-1])
	y_.append(y[-1] + lam * x[-1])

	# пересчет x_k
	tmp    = x_[-1] - lam * y_[-1] + lam * y[-1]
	y.append(y_[-1] + lam * x_[-1] - lam * x[-1])
	x.append(tmp)

plt.scatter(x, y, marker='*', color='orange')
# plt.scatter(x_, y_, marker='*', color='green')

plt.show()
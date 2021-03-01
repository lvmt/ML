#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time 2021/3/1 21:18
# @File 梯度下降.py


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def f(x):
    return x**2


def h(x):
    return 2 * x


X = []
Y = []

x = 2
step = 0.8
f_change = f(x)
f_current = f(x)
X.append(x)
Y.append(f_current)
while f_change > np.e**-10:
    x = x - step * h(x)
    tmp = f(x)
    f_change = np.abs(f_current - tmp)
    f_current = tmp
    X.append(x)
    Y.append(f_current)
print(X)
print(Y)
print(x, f_current)
fig = plt.figure()
a = np.arange(-2.15, 2.15, 0.05)
b = a**2
plt.plot(a, b)
plt.plot(X, Y, "ro--")
plt.show()

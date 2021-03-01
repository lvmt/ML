#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time 2021/3/1 22:21
# @File 多项式.py


# -*- conding:utf-8 -*-
# 准确率：print("电流预测准确率: ", lr2.score(X2_test,Y2_test))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 加载数据
path = "household_power_consumption_1000.txt"
df = pd.read_csv(path, sep=";")
# 数据处理，包括，清除空数据
df1 = df.replace("?", np.nan)
data = df1.dropna(axis=0, how="any")
# 把数据中的字符串转化为数字


def data_formate(x):
    t = time.strptime(' '.join(x), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


X = data.iloc[:, 0:2]
x = X.apply(lambda x: pd.Series(data_formate(x)), axis=1)
y = data.iloc[:, 4]
# 数据分集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
# 标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
# 模型训练
pol = PolynomialFeatures(degree=9)
xtrain_pol = pol.fit_transform(x_train)
xtest_pol = pol.fit_transform(x_test)
lr = LinearRegression()
lr.fit(xtrain_pol, y_train)
y_pridect = lr.predict(xtest_pol)
# 输出参数
print("模型的系数(θ):", lr.coef_)
print("模型的截距:", lr.intercept_)
# print("训练集上R2:",lr.score(x_train, y_train))
print("测试集上R2:", lr.score(xtest_pol, y_test))
mse = np.average((y_pridect - y_test)**2)
rmse = np.sqrt(mse)
print("rmse:", rmse)
# 画图
t = np.arange(len(y_test))
plt.figure(facecolor='w')  # 建一个画布，facecolor是背景色
plt.plot(t, y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_pridect, 'g-', linewidth=2, label='预测值')
plt.legend(loc='upper left')  # 显示图例，设置图例的位置
plt.title("线性回归预测时间和电压之间的关系", fontsize=20)
plt.grid(b=True)  # 加网格
plt.show()

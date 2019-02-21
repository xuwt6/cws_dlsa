# -*- coding:utf-8 -*-
"""
@author:Wentao Xu
@file:draw_epoches.py
@time:2018-12-1023:31
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5AGG')
import matplotlib.pyplot as plt
row_num = 5
data_pku = [95.1, 94.9, 94.9, 94.8, 94.9]
data_msr = [96.9, 96.8, 96.8, 96.9, 96.8]
data_as = [94.9, 94.4, 94.3, 94.1, 94.2]
data_sxu = [94.1, 93.9, 93.9, 93.5, 94]
x_data = [2, 3, 4, 5, 6]

plt.plot()
plt.title('F$_1$ scores for our method with different frequency thresholds')
plt.plot(x_data, data_pku, 'o--', color = 'r')
plt.plot(x_data, data_msr, '*-', color = 'b')
plt.plot(x_data, data_as, 's-', color = 'g')
plt.plot(x_data, data_sxu, '^--', color = 'm')
# plt.plot(x_data, data_sxu, '^:', color = 'w')
plt.ylim(93.2, 98.3)
# plt.xlim(1.75, 6.25)
plt.ylabel('F$_1$ score (%)')
plt.legend(['PKU', 'MSR', 'AS', 'SXU'], loc=1)
plt.xticks(x_data)
plt.xlabel('Frequency threshold')
plt.show()

# -*- coding:utf-8 -*-
"""
@author:Wentao Xu
@file:draw_epochs.py
@time:2018-12-1023:31
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5AGG')
import matplotlib.pyplot as plt
pd_data = pd.read_excel('./results_ours_vs_cai.xls')
row_num = 20
data = np.array(pd_data.values[0:row_num, :])
print(data)
x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# plt.subplot(311)
# plt.title('Performance of different methods on the AS data')
# plt.plot(x_data, data[:, 18], 'o--', color = 'b')
# plt.plot(x_data, data[:, 21], 'o-', color = 'g')
# plt.ylim(80, 100)
# plt.xlim(0, 21)
# plt.ylabel('F$_1$ (%)')
# plt.xticks(x_data)
# plt.legend(['F$_1$ obtained by (Cai 2017)', 'F$_1$ obtained by our method'], loc=4)
# plt.subplot(312)
# plt.plot(x_data, data[:, 19], '*--', color = 'b')
# plt.plot(x_data, data[:, 22], '*-', color = 'g')
# plt.ylim(80, 100)
# plt.xlim(0, 21)
# plt.xticks(x_data)
# plt.ylabel('Precision (%)')
# plt.legend(['Precision obtained by (Cai 2017)', 'Precision obtained by our method'], loc=4)
# plt.subplot(313)
# plt.plot(x_data, data[:, 20], '^--', color = 'b')
# plt.plot(x_data, data[:, 23], '^-', color = 'g')
# plt.ylim(80, 100)
# plt.xlim(0, 21)
# plt.xticks(x_data)
# plt.ylabel('Recall (%)')
# plt.xlabel('Epoches')
# plt.legend(['Recall obtained by (Cai 2017)', 'Recall obtained by our method'], loc=4)
# plt.show()

plt.plot()
# plt.title('Performance of our method on the PKU dataset')
plt.plot(x_data, data[:, 3], 'o-', color = 'r')
plt.plot(x_data, data[:, 4], '*--', color = 'b')
plt.plot(x_data, data[:, 5], '^:', color = 'g')
plt.ylim(80, 100)
plt.xlim(0, 21)
plt.ylabel('Value (%)')
plt.legend(['F$_1$', 'Precision', 'Recall'], loc=4)
plt.xticks(x_data)
plt.xlabel('Epoch')
plt.show()

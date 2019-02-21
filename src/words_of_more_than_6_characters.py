# -*- coding:utf-8 -*-
"""
@author:Wentao Xu
@file:words of more than 6 characters.py
@time:2018/4/2217:22
"""
import re
input_file = '../sxu_data/train.txt'
f = open(input_file, encoding='utf-8')
text_ori = f.readline()
str_ori = ''
len_pass_6 = dict()
while text_ori:
    for chars in text_ori:
        if not re.match('\W', chars, flags=re.U):
            str_ori = str_ori + chars
        else:
            if len(str_ori) > 6:
                # print('%s len = %d' %(str_ori, len(str_ori)))
                try:
                    len_pass_6[str_ori] = len_pass_6[str_ori] + 1
                except:
                    len_pass_6[str_ori] = 1
            str_ori = ''
    text_ori = f.readline()
count_tol = 0
for i in len_pass_6:
    print('%s len = %d count = %d' % (i, len(i), len_pass_6[i]))
    count_tol = count_tol + len_pass_6[i]
print('Number of Non-repeating words of more than 6 characters is %d' % len(len_pass_6))
print('Number of words of more than 6 characters is %d' % count_tol)
f.close()

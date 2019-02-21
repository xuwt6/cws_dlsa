# -*- coding:utf-8 -*-
"""
@author:Wentao Xu
@file:get_the_longest_word.py
@time:2018/5/2220:22
"""
input_file = '../sxu_data/train.txt'
import re

f = open(input_file, encoding='utf-8')
text_ori = f.readline()
str_long_prev = ''
j = 0
j_pre = 0
str_ori = ''
while text_ori:
    for chars in text_ori:
        if not re.match('\W', chars, flags=re.U):
            j = j + 1
            str_ori = str_ori + chars
        else:
            if(len(str_long_prev) < len(str_ori)):
                    str_long_prev = str_ori
                    print('present long word is %s, length = %d' % (str_long_prev, len(str_long_prev)))
                    j_pre = j
            str_ori = ''
            j = 0
    text_ori = f.readline()
print('The longest word of %d characters, and the word in this data set is %s' % (j_pre, str_long_prev))
f.close()
text_ori_sa = []

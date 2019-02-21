# -*- coding:utf-8 -*-
"""
@author:Wentao Xu
@file:count_characters_words.py
@time:2018-12-2423:39
"""
train_file = '../sxu_data/train.txt'
gold_file = '../sxu_data/test.txt'
train_input = open(train_file, encoding='utf-8')
gold_input = open(gold_file, encoding='utf-8')
characters_num = 0
words_num = 0
for line in train_input.readlines():
    for word in line.split():
        words_num = words_num + 1
        for char in word:
            characters_num = characters_num + 1
train_input.close()
print('Number of words in the training file is %d' % words_num)
print('Number of characters in the training file is %d' % characters_num)

num_words_gold = 0
num_chars_gold = 0

for line in gold_input.readlines():
    for word in line.split():
        num_words_gold = num_words_gold + 1
        for char in word:
            num_chars_gold = num_chars_gold + 1
gold_input.close()
print('Number of words in the gold file is %d' % num_words_gold)
print('Number of characters in the gold file is %d' % num_chars_gold)
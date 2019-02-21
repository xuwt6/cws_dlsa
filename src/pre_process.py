# -*- coding: UTF-8 -*-
"""
@author:Wentao Xu
@file:pre_process.py
"""
import re
import numpy as np


def remove_training_space(file_training, file_training_nospace):
    file_training_input = open(file_training, encoding='utf-8')
    file_training_output = open(file_training_nospace, 'w')
    num_line = 0
    for line in file_training_input.readlines():
        num_line = num_line + 1
        sent_output = []
        for word in line.split():
            word = strQ2B(word)
            sent_output.append(word)
        sent_output = ''.join(sent_output)
        sent_output = sent_output + '\r\n'
        sent_output = num_eng_convet_to_symbol(sent_output, re_num, re_eng)
        file_training_output.write(sent_output)
        if num_line % 1000 == 0:
            print('---------------', num_line, '--------------------')
            print('original_line:', line)
            print('revoming_space_outp:', sent_output)
    file_training_output.write('\n')
    file_training_input.close()
    file_training_output.close()


re_num = u'-?\d+\.?\d*%?'
re_eng = u'[A-Za-z_]+'
re_symbol = u'[,\.;?!。“”‘’:()、《》"\']+'


def strB2Q(ustring):
    """Turn half-characters into full-characters"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # half-character blank
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:
            inside_code += 65248
        rstring += chr(inside_code)
    return rstring


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def num_eng_convet_to_symbol(line, re_num, re_eng):
    # symbol_num = re.findall(re_num, line, flags=re.U)
    # symbol_eng = re.findall(re_eng, line, flags=re.U)
    line = re.sub(re_eng, u'X', line, flags=re.U)
    line = re.sub(re_num, u'0', line, flags=re.U)
    return line


def file_convet_utf8(file_in: str, file_out: str) -> object:
    # rNUM = u'[(-|+)?\d+((\.|·)\d+)?%?]'
    word_count, char_count, sent_count = 0, 0, 0
    f_out = ''
    try:
        f_out = open(file_out, 'w')
    except:
        f_out.close()
    with open(file_in, 'r', encoding="utf-8") as f_in:
        for line in f_in.readlines():
            line = strQ2B(line)
            line = num_eng_convet_to_symbol(line, re_num, re_eng)
            new_sent = []
            sent = line.split()
            for word in sent:
                # word = re.sub(u'\s+', '', word, flags=re.U)
                new_sent.append(word)
                char_count += len(word)
                word_count += 1
            sent_count += 1
            f_out.write('  '.join(new_sent) + '\r\n')
        f_out.close()
    print('%s has %d sentences, %d words, %d characters' % (file_in, sent_count, word_count, char_count))


def file_nospace_convet_utf8(file_in: str, file_out: str) -> object:
    # rNUM = u'[(-|+)?\d+((\.|·)\d+)?%?]'
    word_count, char_count, sent_count = 0, 0, 0
    try:
        f_out = open(file_out, 'w')
    except:
        f_out.close()
    with open(file_in, 'r', encoding="utf-8") as f_in:
        for line in f_in.readlines():
            line = strQ2B(line)
            line = num_eng_convet_to_symbol(line)
            new_sent = []
            sent = line.split()
            for word in sent:
                # word = re.sub(u'\s+', '', word, flags=re.U)
                new_sent.append(word)
                char_count += len(word)
                word_count += 1
            sent_count += 1
            f_out.write(''.join(new_sent) + '\r\n')
        f_out.close()
    print('%s has %d sentences, %d words, %d characters' % (file_in, sent_count, word_count, char_count))


def get_threshold(train_file_utf8, proportion):
    word_dict = {}
    length = np.zeros(100, np.int)
    for line in open(train_file_utf8, encoding="utf-8").readlines():
        sent = line.split()
        for word in sent:
            if not re.match(re_symbol, word, flags=re.U):
                try:
                    word_dict[word] = word_dict[word] + 1
                    length[len(word)] = length[len(word)] + 1
                except:
                    word_dict[word] = 1
                    length[len(word)] = length[len(word)] + 1
    occurance = np.zeros(len(word_dict))

    sum_length = np.sum(length)
    proportion_length = int(np.multiply(sum_length, proportion))
    count_length = 0
    i_index = 0
    while count_length <= proportion_length:
        count_length = count_length + length[i_index]
        i_index = i_index + 1
    length_threshold = i_index

    count_occurrence = 0
    i_index = 0
    for key, value in word_dict.items():
        if len(key) == length_threshold:
            occurance[i_index] = value
            i_index = i_index + 1
    occurance = np.sort(-occurance)
    occurance = -occurance
    count_sum = 0
    zero_num = 0.0
    for nozero in occurance:
        if nozero != zero_num:
            count_sum = count_sum + 1
    proportion_occurance = int(np.multiply(count_sum, proportion))
    occurance_threshold = occurance[proportion_occurance]
    return length_threshold, occurance_threshold


def remove_test_space(file_gold, file_test):
    file_input = open(file_gold, encoding='utf-8')
    file_output = open(file_test, 'w')
    num_line = 0
    for line in file_input.readlines():
        num_line = num_line + 1
        sent_output = []
        for word in line.split():
            sent_output.append(word)
        sent_output = ''.join(sent_output)
        sent_output = sent_output + '\r\n'
        file_output.write(sent_output)
        if num_line % 1000 == 0:
            print('---------------', num_line, '--------------------')
            print('test:', line)
            print('output no space:', sent_output)
    file_output.write('\n')
    file_input.close()
    file_output.close()

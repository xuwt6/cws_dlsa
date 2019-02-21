# -*- coding: UTF-8 -*-
import pickle, os, re
from collections import defaultdict
import numpy as np
from src.pre_process import re_symbol
import codecs


def words_convert_chars(file_name):
    file = codecs.open(file_name, 'r', 'utf-8')
    line = file.readline()
    words2char = []
    while line:
        for i in line:
            for j in i.encode('utf-8'):
                words2char.append(j)
        line = file.readline()
    file.close()
    return words2char


def sa_to_num(folder_path, file_training_nospace, sa_training_nospace):
    SA = []
    i, j, temp = 0, 0, 0
    if not os.path.exists(sa_training_nospace):
        os.system('chmod 777 %s/produce_SA' % folder_path)
        os.system('%s/produce_SA %s' % (folder_path, file_training_nospace))
    sa_file = codecs.open(sa_training_nospace, 'rb')
    text_byte = sa_file.read()
    for char_value in text_byte:
        if (i + 1) % 5 == 1:
            temp = char_value & 0xff
        if (i + 1) % 5 == 2:
            temp = (char_value & 0xff) * 256 + temp
        if (i + 1) % 5 == 3:
            temp = (char_value & 0xff) * 256 * 256 + temp
        if (i + 1) % 5 == 4:
            temp = (char_value & 0xff) * 256 * 256 * 256 + temp
        if (i + 1) % 5 == 0:
            temp = (char_value & 0xff) * 256 * 256 * 256 * 256 + temp
            SA.append(temp)
            j = j + 1
            temp = 0
        i = i + 1
    sa_file.close()
    return SA


def initCemb(ndims, file_training, thr=3.):
    f = open(file_training, encoding="utf-8")
    train_vocab = defaultdict(float)
    for line in f.readlines():
        sent = line.split()
        for word in sent:
            for character in word:
                train_vocab[character] += 1
    f.close()
    character_vecs = {}
    for character in train_vocab:
        if train_vocab[character] < thr:
            continue
        character_vecs[character] = np.random.uniform(-0.5 / ndims, 0.5 / ndims, ndims)
    Cemb = np.zeros(shape=(len(character_vecs) + 1, ndims))
    idx = 1
    character_idx_map = dict()
    for character in character_vecs:
        Cemb[idx] = character_vecs[character]
        character_idx_map[character] = idx
        idx += 1
    return Cemb, character_idx_map


def SMEB(lens):
    idxs = []
    for len in lens:
        for i in range(len - 1):
            idxs.append(0)
        idxs.append(len)
    return idxs


def prepare_train_data(folder_path_utf8, character_idx_map, file_training_utf8, file_training_nospace,
                       fragments, sa_training_nospace, threshold_length, threshold_occurence):
    seqs, wlenss, idxss = [], [], []
    words2char = words_convert_chars(file_training_nospace)
    SA = sa_to_num(folder_path_utf8, file_training_nospace, sa_training_nospace)
    sa_len = len(words2char)
    rright = sa_len - 1
    def lower_bound(left, right, p, p_len, SA, words2char):
        while left <= right:
            temp_str_value = []
            mid = int((left + right) / 2)
            j = 0
            j_first = 0
            if (SA[mid] + p_len) <= rright:
                while j_first < p_len:
                    temp_str_value.append(words2char[SA[mid] + j_first])
                    j_first = j_first + 1
            else:
                while j < p_len:
                    if (SA[mid] + j) <= rright:
                        temp_str_value.append(words2char[SA[mid] + j])
                    j = j + 1
            temp_mid = np.array(temp_str_value)
            ss = bytes()
            len_temp_str_value = len(temp_mid)
            i = 0
            while i < len_temp_str_value:
                ss = ss + bytes([temp_mid[i]])
                i = i + 1
            if ss < p:
                left = mid + 1
                continue
            if (mid == 0) and (ss == p):
                return 0
            j = 0
            temp_mid_2 = []
            if mid > 0:
                while j < p_len:
                    temp_mid_2.append(words2char[SA[mid - 1] + j])
                    j = j + 1
            else:
                while j < p_len:
                    j = j + 1
            ss_2 = bytes()
            len_temp_str_value = len(temp_mid_2)
            temp_mid_22 = np.array(temp_mid_2)
            while i < len_temp_str_value:
                ss_2 = ss_2 + bytes([temp_mid_22[i]])
                i = i + 1
            if ss == p and ss_2 == p:
                right = mid - 1
                continue
            if ss == p and ss_2 < p:
                return mid
        return -1

    def upper_bound(left, right, p, p_len, SA, words2char):
        while left <= right:
            temp_mid = []
            temp_mid_2 = []
            j = 0
            mid = int((left + right) / 2)
            if (SA[mid] + p_len) <= rright:
                while j < p_len:
                    temp_mid.append(words2char[SA[mid] + j])
                    j = j + 1
            else:
                while j < p_len:
                    if (SA[mid] + j) <= rright:
                        temp_mid.append(words2char[SA[mid] + j])
                    j = j + 1
            temp_mid_array = np.array(temp_mid)
            temp_mid_array_len = len(temp_mid_array)
            ss = bytes()
            i = 0
            while i < temp_mid_array_len:
                ss = ss + bytes([temp_mid_array[i]])
                i = i + 1
            if ss > p:
                right = mid - 1
                continue
            if (mid == rright) and (ss == p):
                return mid
            j = 0

            if ((SA[mid + 1] + j) <= rright):
                while j < p_len:
                    temp_mid_2.append(words2char[SA[mid + 1] + j])
                    j = j + 1
            else:
                while j < p_len:
                    if ((SA[mid + 1] + j) <= rright):
                        temp_mid_2.append(words2char[SA[mid + 1] + j])
                    j = j + 1
            temp_mid_array_2 = np.array(temp_mid_2)
            temp_mid_array_2_len = len(temp_mid_array_2)
            ss_2 = bytes()
            i = 0
            while i < temp_mid_array_2_len:
                ss_2 = ss_2 + bytes([temp_mid_array_2[i]])
                i = i + 1
            if (ss == p) and (ss_2 == p):
                left = mid + 1
                continue
            if (ss == p) and (ss_2 > p):
                return mid
        return -1

    def search_lower_upper_bound(temp_search):
        left = 0
        right = rright
        p = temp_search.encode(encoding="utf-8")
        p_len = len(p)
        while left <= right:
            temp_str = []
            k = 0
            k_first = 0
            mid = int((left + right) / 2)

            if ((SA[mid] + p_len) <= rright):
                while k_first < p_len:
                    temp_str.append(words2char[SA[mid] + k_first])
                    k_first = k_first + 1
            else:
                while k < p_len:
                    if ((SA[mid] + k) <= rright):
                        temp_str.append(words2char[SA[mid] + k])
                    k = k + 1
            temp_str_value = np.array(temp_str)
            i = 0
            ss = bytes()
            len_temp_str_value = len(temp_str_value)
            while i < len_temp_str_value:
                ss = ss + bytes([temp_str_value[i]])
                i = i + 1
            if ss < p:
                left = mid + 1
                continue
            if ss > p:
                right = mid - 1
                continue
            if ss == p:
                result_0 = lower_bound(0, mid, p, p_len, SA, words2char)
                result_1 = upper_bound(mid, rright, p, p_len, SA, words2char)
                z = 0
                fench_data = []
                ss_3 = bytes()
                while z < p_len:
                    fench_data.append(words2char[SA[result_0] + z])
                    z = z + 1
                fench_data_array = np.array(fench_data)
                fench_data_array_len = len(fench_data_array)
                i = 0
                while i < fench_data_array_len:
                    ss_3 = ss_3 + bytes([fench_data_array[i]])
                    i = i + 1
                # print(ss_3.decode())
                if result_0 == -1:
                    print("The total number of [%s] is %d and the length of search pattern is %d" % (p, 0, p_len))
                    return 0
                else:
                    return result_1 - result_0 + 1
        return 0
    fragments_num = dict()
    line_num = 0
    for line in open(file_training_utf8, encoding="utf-8").readlines():
        line_num += 1
        # print('The training line: ', '%d: %s' % (line_num, line), end='')
        sent = line.split()
        left = 0
        for idx, word in enumerate(sent):
            if re.match(re_symbol, word, flags=re.U):
                while idx > left:
                    len_words = [len(i) for i in sent[left: idx]]
                    sent_list = sent[left: idx]
                    ri = len(len_words) - 1
                    li = 0
                    while li <= ri:
                        temp_sent = sent_list[li: ri + 1]
                        temp_str = ''.join(temp_sent)
                        len_temp_str = len(temp_str)
                        if len_temp_str >= threshold_length:
                            try:
                                count_sent_str = fragments_num[temp_str]
                            except:
                                try:
                                    count_sent_str = search_lower_upper_bound(temp_str)
                                except:
                                    count_sent_str = 0
                            if count_sent_str >= threshold_occurence:
                                seqs.append(list(''.join(temp_str)))
                                wlenss.append([len(word) for word in temp_sent])
                                # print(temp_sent, temp_str)
                                left = left + len(temp_sent)
                                fragments_num[temp_str] = count_sent_str
                                break
                            else:
                                if li == ri:
                                    seqs.append(list(''.join(temp_str)))
                                    wlenss.append([len(word) for word in temp_sent])
                                    left = left + 1
                                    # print('ri = li', temp_sent, temp_str)
                                ri = ri - 1
                        else:
                            temp_sent = sent_list[li:]
                            temp_str = ''.join(temp_sent)
                            seqs.append(list(''.join(temp_str)))
                            wlenss.append([len(word) for word in temp_sent])
                            # print(temp_sent, temp_str)
                            left = left + len(temp_sent)
                            break
                left = left + 1

        if left != len(sent):
            while idx >= left:
                len_words = [len(i) for i in sent[left:]]
                sent_list = sent[left:]
                ri = len(len_words) - 1
                li = 0
                while li <= ri:
                    temp_sent = sent_list[li: ri + 1]
                    temp_str = ''.join(temp_sent)
                    len_temp_str = len(temp_str)
                    if len_temp_str >= threshold_length:
                        try:
                            count_sent_str = fragments_num[temp_str]
                        except:
                            try:
                                count_sent_str = search_lower_upper_bound(temp_str)
                            except:
                                count_sent_str = 0
                        if count_sent_str >= threshold_occurence:
                            seqs.append(list(''.join(temp_str)))
                            wlenss.append([len(word) for word in temp_sent])
                            # print(temp_sent, temp_str)
                            fragments_num[temp_str] = count_sent_str
                            left = left + len(temp_sent)
                            break
                        else:
                            if ri == li:
                                seqs.append(list(''.join(temp_str)))
                                wlenss.append([len(word) for word in temp_sent])
                                left = left + 1
                                # print('left != idx, ri = li', temp_sent, temp_str)
                            ri = ri - 1
                    else:
                        temp_sent = sent_list[li:]
                        temp_str = ''.join(temp_sent)
                        seqs.append(list(''.join(temp_str)))
                        wlenss.append([len(word) for word in temp_sent])
                        # print('left != idx', temp_sent, temp_str)
                        left = left + len(temp_sent)
                        break

    seqs = [[character_idx_map[character] if character in character_idx_map else 0 for character in seq] for seq in seqs]
    f_pkl = open(fragments, 'wb')
    pickle.dump(fragments_num, f_pkl)
    f_pkl.close()
    # for i in fragments_num:
        # print('train', i, fragments_num[i])
    for w_lens in wlenss:
        idxss.append(SMEB(w_lens))
    return seqs, wlenss, idxss

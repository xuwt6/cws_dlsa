# -*- coding: UTF-8 -*-
import re, pickle, os
from src.pre_process import strQ2B, num_eng_convet_to_symbol, strB2Q
from src.pre_process import re_eng, re_num
from src.pre_process import re_symbol
from src.tools import words_convert_chars
from src.tools import sa_to_num


def test_suffix_array(cws, folder_path_utf8, file_training_nospace, sa_training_nospace, file_test, fragments,
                      threshold_length_test, threshold_occurence_test, output_path):
    character_idx_map = cws.character_idx_map
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
                    # print("The total number of [%s] is %ld and the length of search pattern is %d" %(p, (result_1 - result_0 + 1), p_len))
                    return result_1 - result_0 + 1
        return 0


    pkl_read = open(fragments, 'rb')
    fragments_num = pickle.load(pkl_read)

    def seg(char_seq, text):
        lens = cws.forward(char_seq)
        res, begin = [], 0
        for wlen in lens:
            res.append(''.join(text[begin:begin + wlen]))
            begin += wlen
        return res

    fo = None
    if os.path.exists('../result'):
        os.system('chmod 777 ../result')
        fo = open(output_path, 'w')
    else:
        os.system('mkdir ../result')
        os.system('chmod 777 ../result')
        fo = open(output_path, 'w')
    line_num = 0
    for line in open(file_test, encoding="utf-8").readlines():
        # line = ',,瓦西  里斯的船 只中有４０％驶向i love you, which-is a good_like远东，每个月几乎都有两三条船停靠中国港口。'
        output = []
        line_utf8 = strQ2B(line)
        list_num = re.findall(re_num, line_utf8, flags=re.U)
        list_eng = re.findall(re_eng, line_utf8, flags=re.U)
        line_utf8 = num_eng_convet_to_symbol(line_utf8, re_num, re_eng)
        line_num = line_num + 1
        sent = ' '.join(line_utf8)
        sent = sent.split()
        left = 0
        output_sent = []
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
                        if len_temp_str >= threshold_length_test:
                            try:
                                count_sent_str = fragments_num[temp_str]
                            except:
                                try:
                                    count_sent_str = search_lower_upper_bound(temp_str)
                                except:
                                    count_sent_str = 0
                            if count_sent_str >= threshold_occurence_test:
                                seqs = list(''.join(temp_str))
                                seqs = [character_idx_map[character] if character in character_idx_map else 0 for character in seqs]
                                words = seg(seqs, list(''.join(temp_str)))
                                output_sent.extend(words)
                                left = left + len(temp_sent)
                                fragments_num[temp_str] = count_sent_str
                                break
                            else:
                                ri = ri - 1
                        else:
                            sent_end = sent_list[li:]
                            seqs = list(''.join(sent_end))
                            seqs = [character_idx_map[character] if character in character_idx_map else 0 for character in seqs]
                            words = seg(seqs, list(''.join(sent_end)))
                            output_sent.extend(words)
                            left = left + len(sent_end)
                            break
                left = left + 1
                output_sent.append(word)

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
                    if len_temp_str >= threshold_length_test:
                        try:
                            count_sent_str = fragments_num[temp_str]
                        except:
                            try:
                                count_sent_str = search_lower_upper_bound(temp_str)
                            except:
                                count_sent_str = 0
                        if count_sent_str >= threshold_occurence_test:
                            seqs = list(''.join(temp_str))
                            seqs = [character_idx_map[character] if character in character_idx_map else 0 for character in seqs]
                            words = seg(seqs, list(''.join(temp_str)))
                            output_sent.extend(words)
                            left = left + len(temp_sent)
                            fragments_num[temp_str] = count_sent_str
                            break
                        else:
                            ri = ri - 1
                    else:
                        sent_end = sent_list[li:]
                        seqs = list(''.join(sent_end))
                        seqs = [character_idx_map[character] if character in character_idx_map else 0 for character in
                                seqs]
                        words = seg(seqs, list(''.join(sent_end)))
                        output_sent.extend(words)
                        left = left + len(sent_end)
                        break

        i_list, j_list = 0, 0

        for idx, word in enumerate(output_sent):
            word = list(word)
            for i, character in enumerate(word):
                if character == '0':
                    word[i] = list_num[i_list]
                    i_list = i_list + 1
                elif character == 'X':
                    word[i] = list_eng[j_list]
                    j_list = j_list + 1
                else:
                    word[i] = character
            output.extend([''.join(word)])
        char_num = 0
        output_ori = []
        len_line = len(line)
        for words in output:
            words_str = ''
            for _ in words:
                if len_line > char_num:
                    char_ = line[char_num]
                    words_str = words_str + char_
                    char_num = char_num + 1
            output_ori.extend([words_str])
        output = '  '.join(output_ori) + '\r\n'
        # output_utf =
        # if line_num % 300 == 0:
        #     print('---------', line_num, '---------')
        #     print('utf8_b:       ', line_utf8, end='')
        #     print('utf8_ori:     ', line, end='')
        #     print('utf8_turn_ori:', output, end='')
        fo.write(output)
    fo.close()

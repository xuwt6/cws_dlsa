# -*- coding: UTF-8 -*-
import time, os
from collections import Counter, namedtuple
from src.score import os_system
import numpy as np
import dynet as dy
from src.tools import initCemb, prepare_train_data
from src.test import test_suffix_array
from src.pre_process import remove_training_space, file_convet_utf8, file_nospace_convet_utf8
from src.pre_process import remove_test_space

np.random.seed(970)
Sentence = namedtuple('Sentence', ['score', 'score_expr', 'LSTMState', 'y', 'prevState', 'wlen', 'golden'])


class CWS(object):
    def __init__(self, Cemb, character_idx_map, options):
        model = dy.Model()
        self.trainer = dy.MomentumSGDTrainer(model, options['lr'], options['momentum'])
        # self.trainer = dy.AdagradTrainer(model, options['lr'])
        self.params = self.initParams(model, Cemb, options)
        self.options = options
        self.model = model
        self.character_idx_map = character_idx_map
        self.known_words = None

    def load(self, filename):
        # self.model.load(filename)
        pass

    def save(self, filename):
        self.model.save(filename)

    def use_word_embed(self, known_words):
        self.known_words = known_words
        self.params['word_embed'] = self.model.add_lookup_parameters((len(known_words), self.options['word_dims']))

    def initParams(self, model, Cemb, options):
        # initialize the model parameters
        params = dict()
        params['embed'] = model.add_lookup_parameters(Cemb.shape)
        for row_num, vec in enumerate(Cemb):
            params['embed'].init_row(row_num, vec)
        # word_dims=100, nhiddens=50,
        params['lstm'] = dy.LSTMBuilder(1, options['word_dims'], options['nhiddens'], model)
        params['reset_gate_W'] = []
        params['reset_gate_b'] = []
        params['com_W'] = []
        params['com_b'] = []
        params['word_score_U'] = model.add_parameters(options['word_dims'])
        params['predict_W'] = model.add_parameters((options['word_dims'], options['nhiddens']))
        params['predict_b'] = model.add_parameters(options['word_dims'])
        for wlen in range(1, options['max_word_len'] + 1):
            params['reset_gate_W'].append(
                model.add_parameters((wlen * options['char_dims'], wlen * options['char_dims'])))
            params['reset_gate_b'].append(model.add_parameters(wlen * options['char_dims']))
            params['com_W'].append(model.add_parameters((options['word_dims'], wlen * options['char_dims'])))
            params['com_b'].append(model.add_parameters(options['word_dims']))
        params['<BoS>'] = model.add_parameters(options['word_dims'])
        return params

    def renew_cg(self):
        # renew the compute graph for every single instance
        dy.renew_cg()

        param_exprs = dict()
        param_exprs['U'] = dy.parameter(self.params['word_score_U'])
        param_exprs['pW'] = dy.parameter(self.params['predict_W'])
        param_exprs['pb'] = dy.parameter(self.params['predict_b'])
        param_exprs['<bos>'] = dy.parameter(self.params['<BoS>'])
        self.param_exprs = param_exprs

    def word_repr(self, char_seq, cembs):
        # obtain the word representation when given its character sequence
        wlen = len(char_seq)
        if 'rgW%d' % wlen not in self.param_exprs:
            self.param_exprs['rgW%d' % wlen] = dy.parameter(self.params['reset_gate_W'][wlen - 1])
            self.param_exprs['rgb%d' % wlen] = dy.parameter(self.params['reset_gate_b'][wlen - 1])
            self.param_exprs['cW%d' % wlen] = dy.parameter(self.params['com_W'][wlen - 1])
            self.param_exprs['cb%d' % wlen] = dy.parameter(self.params['com_b'][wlen - 1])
        chars = dy.concatenate(cembs)
        reset_gate = dy.logistic(self.param_exprs['rgW%d' % wlen] * chars + self.param_exprs['rgb%d' % wlen])
        word = dy.tanh(self.param_exprs['cW%d' % wlen] * dy.cmult(reset_gate, chars) + self.param_exprs['cb%d' % wlen])
        if self.known_words is not None and tuple(char_seq) in self.known_words:
            return (word + dy.lookup(self.params['word_embed'], self.known_words[tuple(char_seq)])) / 2.
        return word

    def greedy_search(self, char_seq, truth=None, mu=0.):
        global golden_sent
        init_state = self.params['lstm'].initial_state().add_input(self.param_exprs['<bos>'])
        init_y = dy.tanh(self.param_exprs['pW'] * init_state.output() + self.param_exprs['pb'])
        init_score = dy.scalarInput(0.)  # expression 24/1
        init_sentence = Sentence(score=init_score.scalar_value(), score_expr=init_score, LSTMState=init_state, y=init_y,
                                 prevState=None, wlen=None, golden=True)
        if truth is not None:
            cembs = [dy.dropout(dy.lookup(self.params['embed'], char), self.options['dropout_rate']) for char in
                     char_seq]
        else:
            cembs = [dy.lookup(self.params['embed'], char) for char in char_seq]

        start_agenda = init_sentence  # Sentence(score=0.0, score_expr=expression 23/1, LSTMState=<_dynet.RNNState
        agenda = [start_agenda]  # [Sentence(score=0.0, score_expr=expression 23/1, LSTMState=<_dynet.RNNState object
        for idx, _ in enumerate(char_seq, 1):  # from left to right, character by character idx的值从1开始，结尾值等于这个句子的长度
            now = None
            for wlen in range(1, (min(idx, self.options[
                'max_word_len']) + 1)):  # generate word candidate vectors wlen 从1开始，到当前处理到的字符结尾
                word = self.word_repr(char_seq[idx - wlen:idx],
                                      cembs[idx - wlen:idx])  # word的值从当前字符，依次往前减1，直到word的值从1到当前字符。
                sent = agenda[idx - wlen]  # sent 取到的值也是从agenda[n - 1]，直到取到agenda[0] 为止

                if truth is not None:
                    word = dy.dropout(word, self.options['dropout_rate'])

                word_score = dy.dot_product(word, self.param_exprs['U'])

                if truth is not None:
                    golden = sent.golden and truth[idx - 1] == wlen  # sent当前为golden，并且判断字符是不是分割点，如果是 就为True.
                    margin = dy.scalarInput(mu * wlen if truth[idx - 1] != wlen else 0.)  # 如果是边界，则为0
                    # print((mu * wlen if truth[idx - 1] != wlen else 0.))
                    score = margin + sent.score_expr + dy.dot_product(sent.y,
                                                                      word) + word_score  # score的值: 边界 + 语句的值 + 语句和当前词语的乘机值 + 当前词语的值
                else:
                    golden = False
                    score = sent.score_expr + dy.dot_product(sent.y, word) + word_score

                good = (now is None or now.score < score.scalar_value())  # 当前的值为空，或者以前的值比现在计算的值要小，即现在得到的值较好。
                if golden or good:
                    new_state = sent.LSTMState.add_input(word)  # 把当前计算的word加入语句中，得到一个新的new_state
                    new_y = dy.tanh(self.param_exprs['pW'] * new_state.output() + self.param_exprs['pb'])  # new_y的值为
                    new_sent = Sentence(score=score.scalar_value(), score_expr=score, LSTMState=new_state, y=new_y,
                                        prevState=sent, wlen=wlen, golden=golden)
                    if good:
                        now = new_sent
                    if golden:
                        golden_sent = new_sent
            agenda.append(now)
            if truth is not None and truth[idx - 1] > 0 and (not now.golden):
                return now.score_expr - golden_sent.score_expr
        if truth is not None:
            return now.score_expr - golden_sent.score_expr
        return agenda

    def forward(self, char_seq):
        self.renew_cg()
        agenda = self.greedy_search(char_seq)
        now = agenda[-1]
        ans = []
        while now.prevState is not None:
            ans.append(now.wlen)
            now = now.prevState
        return reversed(ans)

    def backward(self, char_seq, truth):
        self.renew_cg()
        loss = self.greedy_search(char_seq, truth, self.options['margin_loss_discount'])
        res = loss.scalar_value()
        loss.backward()
        return res


def dy_train_model(
        folder_path: str = '../sxu_data',
        folder_path_utf8: str = '../data_utf8',
        data_name: str = 'sxu',
        max_epochs: int = 20,
        batch_size: int = 100,
        char_dims: int = 100,
        word_dims: int = 50,
        nhiddens: float = 50,
        dropout_rate: float = 0.1,
        margin_loss_discount: float = 0.1,
        max_word_len: float = 50,
        shuffle_data: bool = True,
        lr: float = 0.5,
        momentum: float = 0.5,
        word_proportion: float = 0.7,
        threshold_length: int = 7,
        threshold_occurence: int = 2
) -> object:
    file_training = os.path.join(folder_path, '%s_training.utf8' % data_name)
    # file_training = '../ctb_data/ctb6.train.seg'
    file_training_utf8 = os.path.join(folder_path_utf8, '%s_training.utf8' % data_name)
    # file_training_utf8 = '../data_utf8/ctb6.train.seg.utf8'
    file_training_nospace = os.path.join(folder_path_utf8, '%s_training_nospace.utf8' % data_name)
    # file_training_nospace = '../data_utf8/ctb6.train.seg.utf8.nospace'
    file_gold = os.path.join(folder_path, '%s_test_gold.utf8' % data_name)
    # file_gold = '../ctb_data/ctb6.test.seg'
    file_test = os.path.join(folder_path, '%s_test.utf8' % data_name)
    # file_test = '../ctb_data/ctb6.test.nospace'
    file_test_utf8 = os.path.join(folder_path_utf8, '%s_test.utf8' % data_name)
    # file_test_utf8 = '../data_utf8/ctb6.test.seg.nospace'

    #######################
    # if not os.path.exists(file_test):
    #     remove_test_space(file_gold, file_test)
    ########################

    if not os.path.exists(file_training_nospace):
        remove_training_space(file_training, file_training_nospace)
    if not os.path.exists(file_training_utf8):
        file_convet_utf8(file_training, file_training_utf8)
    if not os.path.exists(file_test_utf8):
        file_convet_utf8(file_test, file_test_utf8)

    fragments = '%s/%s_fragments.pkl' % (folder_path_utf8, data_name)
    # fragments = '../sxu_data/sxu_fragments.pkl'
    # sa_training = '%s.sa5' % file_training_utf8
    sa_training = '%s.sa5' % file_training_utf8
    sa_training_nospace = '%s.sa5' % file_training_nospace
    # threshold_length, threshold_occurence = get_threshold(file_training_utf8, threshold_proportion)
    # print('threshold_length = ', threshold_length, 'threshold_occurence = ', threshold_occurence)
    options = locals().copy()
    print('Model options:')
    for kk, vv in options.items():
        print('\t', kk, '\t', vv)
    Cemb, character_idx_map = initCemb(char_dims, file_training_utf8)
    cws = CWS(Cemb, character_idx_map, options)
    char_seq, _, truth = prepare_train_data(folder_path_utf8,
        cws.character_idx_map, file_training_utf8, file_training_nospace,
        fragments, sa_training_nospace, threshold_length, threshold_occurence)
    print('\n-------------- char_seq of training is ready! ------------------')
    if word_proportion > 0:
        word_counter = Counter()
        for chars, labels in zip(char_seq, truth):
            word_counter.update(tuple(chars[idx - label:idx]) for idx, label in enumerate(labels, 1))
        known_word_count = int(word_proportion * len(word_counter))
        known_words = dict(word_counter.most_common()[:known_word_count])
        idx = 0
        for word in known_words:
            known_words[word] = idx
            idx += 1
        cws.use_word_embed(known_words)
    n = len(char_seq)
    print('Total number of training instances:', n)
    print('Start training model')
    start_time = time.time()
    print(start_time)
    nsamples = 0
    dir_parameter = '%s_dropout_%.3f_margin_loss_dis_%.3f' % (data_name, dropout_rate, margin_loss_discount)
    if not os.path.exists(dir_parameter):
        os.makedirs(dir_parameter)
    for eidx in range(max_epochs):
        idx_list = list(range(n))
        if shuffle_data:
            np.random.shuffle(idx_list)
            print('./' + dir_parameter + '/' + '%d.txt' % (eidx + 1))
        for idx in idx_list:
            loss = cws.backward(char_seq[idx], truth[idx])
            if np.isnan(loss):
                print('somthing went wrong, loss is NAN.')
                return
            nsamples += 1
            if nsamples % batch_size == 0:
                cws.trainer.update()
        cws.trainer.update_epoch(1.)
        end_time = time.time()
        print('Trained %s epoch(s) (%d samples) took %.lfs per epoch' % (
            eidx + 1, nsamples, (end_time - start_time) / (eidx + 1)))
        test_suffix_array(cws, folder_path_utf8, file_training_nospace, sa_training_nospace, file_test, fragments, threshold_length,
                          threshold_occurence, '../result/dev_result%d' % (eidx + 1))
        os_system('%s,%s,%d,%d' % (folder_path, file_gold, eidx + 1, eidx + 1))

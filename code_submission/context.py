#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-03-18
import copy
import random
from multiprocessing import Pool

import numpy as np

from CONSTANT import MAX_VALID_PER_CLASS, MAX_VALID_SET_SIZE, MIN_VALID_PER_CLASS, SPEECH_SIMPLE_MODEL_MAX_SAMPLE_NUM
from features import MEL_SPECTROGRAM, get_specified_feature_func, STFT, MFCC
from stage import Preds
from stages import FastStage, EnhancementStage, ExplorationStage
from tools import is_multilabel, timeit, log
from math import ceil


class Context:
    def __init__(self, metadata, keep_num=5, config=None):
        self._metadata = metadata
        self._keep_num = keep_num
        self._num_classes = self._metadata['class_num']
        self._is_multilabel = False
        self.is_last_stage = False

        self._train_loop_num = 0

        self._raw_x, self._raw_y = None, None
        self._raw_test_x = None
        self._train_x, self._train_y = None, None
        self._train_y_hat, self._train_y_true = None, None
        self._val_x, self._val_y = None, None
        self.train_index, self.val_index = None, None

        self._total_num = None
        self._train_num = None
        self._val_num = None
        self._test_num = None

        self._total_class_index = {}    # use all data
        self._train_class_index = {}  # only use train data without val data
        self._replace_class_index = None    # same as _class_index, but we can remove items from it

        self._cut_length = None
        self._mean_length = None

        self._each_model_best_preds = {}
        self._kbest_preds = [Preds('', '', -1, None)] * self._keep_num
        self._max_score = 0

        self._stage = None
        self.resnet_stage = EnhancementStage(self, 2000, STFT, None)

        self.lr_last_preds = []
        self._config = config
        self._is_finished = False

        self.use_knn_fix = False

    def init_stage(self):
        pre_func = get_specified_feature_func(MEL_SPECTROGRAM, n_mels=30, use_power_db=True)
        end_loop_num = min(3, ceil(self._metadata["train_num"]/(self.num_classes * 3)))
        end_loop_num = min(end_loop_num, ceil(self._metadata["train_num"]/SPEECH_SIMPLE_MODEL_MAX_SAMPLE_NUM))
        end_loop_num = max(end_loop_num, 2)
        self._stage = FastStage(self,
                                end_loop_num,
                                MEL_SPECTROGRAM, pre_func, 8)
        log(f"Fast stage end_loop_num: {end_loop_num}")

    @property
    def raw_data(self):
        return self._raw_x, self._raw_y

    @raw_data.setter
    def raw_data(self, dataset):
        x, y = dataset
        self._is_multilabel = is_multilabel(y)
        self._raw_x, self._raw_y = np.asarray(x, dtype=np.object), np.asarray(y, dtype=np.int32)
        self._cut_length, self._mean_length = self._raw_data_discovery(self._raw_x, self._raw_y)
        train_index, val_index = self._train_test_split()
        self.train_index, self.val_index = train_index, val_index
        self._train_x, self._train_y = self._raw_x[train_index], self._raw_y[train_index]
        self._val_x, self._val_y = self._raw_x[val_index], self._raw_y[val_index]
        self._init_class_index()
        self._total_num, self._train_num, self._val_num =\
            len(self._raw_y), len(train_index), len(val_index)
        log("total_num {} train_num {} val_num {}".format(self._total_num, self._train_num, self._val_num))
        # TODO need class balance?

    @property
    def train_data(self):
        return self._train_x, self._train_y

    @property
    def val_data(self):
        return self._val_x, self._val_y

    @property
    def train_y_true(self):
        return self._train_y_true

    @train_y_true.setter
    def train_y_true(self, y_true):
        self._train_y_true = y_true

    @property
    def train_y_hat(self):
        return self._train_y_hat

    @train_y_hat.setter
    def train_y_hat(self, y_hat):
        self._train_y_hat = y_hat

    @property
    def raw_test_data(self):
        return self._raw_test_x

    @raw_test_data.setter
    def raw_test_data(self, test_x):
        self._raw_test_x = np.asarray(test_x, np.object)
        self._test_num = len(self._raw_test_x)

    @property
    def cut_length(self):
        return self._cut_length

    # @cut_length.setter
    # def cut_length(self, len):
    #     self._cut_length = len

    @property
    def train_loop_num(self):
        return self._train_loop_num

    @train_loop_num.setter
    def train_loop_num(self, value):
        self._train_loop_num = value

    @property
    def class_index(self):
        return self._train_class_index

    @property
    def total_num(self):
        return self._total_num

    @property
    def train_num(self):
        return self._train_num

    @property
    def test_num(self):
        return self._test_num

    @property
    def val_num(self):
        return self._val_num

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def config(self):
        return self._config

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, stg):
        self._stage = stg

    @property
    def mp(self):
        if self._mp is None:
            self._mp = Pool()
        return self._mp

    @property
    def is_finished(self):
        return self._is_finished

    @is_finished.setter
    def is_finished(self, value):
        self._is_finished = value

    @property
    def is_multilabel(self):
        return self._is_multilabel

    @property
    def max_score(self):
        return self._max_score

    @property
    def mean_length(self):
        return self._mean_length

    @timeit
    def _raw_data_discovery(self, x, y, ratio=0.95):
        l = len(y)
        # length discovery
        lens = [len(_) for _ in x]
        lens.sort()
        len_max = lens[-1]
        len_min = lens[0]
        len_specified = lens[int(l * ratio)]
        len_mean = int(sum(lens) / l)
        log(
            f"Max length: {len_max}; Min length {len_min}; {ratio * 100}% length {len_specified} Mean length {len_mean}")
        cut_length = len_specified

        # class discovery
        each_class_num = np.sum(y, axis=0)
        each_class_num.sort()
        class_max = each_class_num[-1]
        class_min = each_class_num[0]
        class_95 = each_class_num[int(len(each_class_num) * 0.95)]
        class_mean = int(sum(each_class_num) / len(each_class_num))
        log(f"Max class: {class_max}; Min class {class_min}; 95% class {class_95} Mean Class {class_mean};"
            f" class distribution {list(each_class_num / l)}")

        return cut_length, len_mean

    def _train_test_split(self, ratio=0.8):
        # TODO train test split for multilables
        if self._is_multilabel:
            all_index, sample_nums = np.arange(len(self._raw_y)).tolist(), len(self._raw_y)
            train_index = random.sample(all_index, int(sample_nums * ratio))
            val_index = list(set(all_index).difference(set(train_index)))
            return train_index, val_index

        train_index, val_index = [], []
        max_val_per_class = min(MAX_VALID_PER_CLASS, MAX_VALID_SET_SIZE // self._num_classes)
        for i in range(self._num_classes):
            self._total_class_index[i] = list(np.where(self._raw_y[:, i] == 1)[0])
        for i in range(self._num_classes):
            tmp = random.sample(self._total_class_index[i],
                                max(MIN_VALID_PER_CLASS, int(len(self._total_class_index[i]) * (1 - ratio))))
            if len(tmp) > max_val_per_class:
                tmp = tmp[:max_val_per_class]
            val_index += tmp
            differ_set = set(self._total_class_index[i]).difference(set(tmp))
            # avoid some classes only have one sample
            if len(differ_set) == 0:
                differ_set = set(tmp)
            train_index += list(differ_set)
        random.shuffle(train_index)
        random.shuffle(val_index)
        # TODO whether random split, not each class
        return train_index, val_index

    def _init_class_index(self):
        for i in range(self._num_classes):
            all_class_index_i = list(np.where(self._raw_y[:, i] == 1)[0])
            self._train_class_index[i] = list(set(all_class_index_i).intersection(self.train_index))
        self._replace_class_index = copy.deepcopy(self._train_class_index)

    def get_samples(self, per_class_sample_num, min_sample_num, max_sample_num=0,
                    replace=True, reset=False, sample_all=True, use_all_data=False):
        if min_sample_num < self.num_classes:
            min_sample_num = self.num_classes
        each_class_sample_num = round(min_sample_num / self._num_classes)
        each_class_sample_num = max(per_class_sample_num, each_class_sample_num)
        each_class_sample_num = max(each_class_sample_num, 1)

        # TODO: For multi-label
        if self.is_multilabel:
            if use_all_data:
                all_index = self.train_index + self.val_index
            else:
                all_index = self.train_index
            multilable_sample_num = each_class_sample_num * self._num_classes
            sample_index = all_index if len(all_index) <= multilable_sample_num \
                else random.sample(all_index, multilable_sample_num)
        else:
            sample_index = []
            for i in range(self._num_classes):
                if replace:
                    if use_all_data:
                        class_index = self._total_class_index[i]
                    else:
                        class_index = self._train_class_index[i]
                    if len(class_index) >= each_class_sample_num:
                        indexes = random.sample(class_index, each_class_sample_num)
                    else:
                        indexes = class_index
                else:
                    if len(self._replace_class_index[i]) >= each_class_sample_num:
                        indexes = random.sample(self._replace_class_index[i], each_class_sample_num)
                    else:
                        indexes = self._replace_class_index[i]
                    tmp = list(set(self._replace_class_index[i]).difference(indexes))
                    self._replace_class_index[i] = tmp
                sample_index += indexes

            if reset:
                self._replace_class_index = copy.deepcopy(self._train_class_index)

        if 0 < max_sample_num < len(sample_index):
            sample_index = random.sample(sample_index, max_sample_num)

        random.shuffle(sample_index)
        return sample_index

    def add_predicts(self, pred):
        if pred.name not in self._each_model_best_preds or \
                self._each_model_best_preds[pred.name].score < pred.score:
            self._each_model_best_preds[pred.name] = pred

        if self._kbest_preds[-1].score < pred.score:
            self._kbest_preds[-1] = pred
            log(f"add {pred} to kbest preds")

        if self._max_score < pred.score:
            log(f"get best preds {pred}, last {self._max_score}")
            self._max_score = pred.score

        self._kbest_preds.sort(key=lambda p: p.score, reverse=True)

    def ensemble_predicts(self):
        ensemble_preds = list(filter(lambda p: p.score >= 0, self._kbest_preds))
        log(f"each model best preds {[str(p) for p in self._each_model_best_preds.values()]};"
            f" ensemble preds {[str(p) for p in ensemble_preds]}")

        ensemble_preds = list(map(lambda p: p.preds, ensemble_preds))

        return np.mean(ensemble_preds, axis=0)

    def get_min_score_in_kbest(self):
        return self._kbest_preds[-1].score

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-03-18
import abc

from past.builtins import cmp


class Preds:
    def __init__(self, model_name, fea_name, score, preds):
        self._model_name = model_name
        self._fea_name = fea_name
        self._score = score
        self._preds = preds

    def __str__(self):
        return self.name + '#' + str(self._score)

    @property
    def model_name(self):
        return self._model_name

    @property
    def fea_name(self):
        return self._fea_name

    @property
    def score(self):
        return self._score

    @property
    def preds(self):
        return self._preds

    @property
    def name(self):
        return self._model_name + '#' + self.fea_name


class Stage(metaclass=abc.ABCMeta):
    def __init__(self, context, end_loop_num, fea_name, pre_func, **kwargs):
        super(Stage, self).__init__()
        self._context = context
        self._fea_name = fea_name
        self._pre_func = pre_func
        self._need_transition = False
        self._stage_name = None

        self._stage_end_loop_num = end_loop_num
        self._stage_loop_num = 0
        self._stage_test_loop_num = 0

        self._pre_train_x_dict = {}
        self._pre_train_x, self._pre_train_y = None, None
        self._pre_val_x, self._pre_val_y = None, None
        self._pre_test_x = None

        self._model = None

    @property
    def ctx(self):
        return self._context

    @ctx.setter
    def ctx(self, value):
        self._context = value

    @abc.abstractmethod
    def _preprocess_data(self):
        self.ctx.train_loop_num += 1
        self._stage_loop_num += 1

    @abc.abstractmethod
    def _transition(self):
        pass

    def train(self, remain_time_budget=None):
        self.ctx.time_budget = remain_time_budget

    def test(self, remain_time_budget=None):
        self.ctx.time_budget = remain_time_budget

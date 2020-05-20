#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os
os.system("pip install future")
os.system("pip install kapre==0.1.4 -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install keras==2.2.4 -i https://pypi.tuna.tsinghua.edu.cn/simple")

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from context import Context
from tools import log, timeit
from test_backend import knn_test_fix

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
sess = tf.Session(config=config)
set_session(sess)


class Model(object):
    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 7,
             "train_num": 428,
             "test_num": 107,
             "time_budget": 1800}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_loop_num = 0
        log(f'Metadata: {self.metadata}')

        self._context = Context(self.metadata)

        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        self._has_exception = False
        self._last_pred = None

    @timeit
    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.

        :param train_dataset: tuple, (train_x, train_y)
            train_x: list of vectors, input train models raw data.
            train_y: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        try:
            if self._context.is_finished:
                log("Finish all stages")
                self.done_training = True
                return
            self.train_loop_num += 1

            if self.train_loop_num == 1:
                self._context.raw_data = train_dataset
                self._context.init_stage()

            self._context.stage.train(remaining_time_budget)
        except Exception as exp:
            log("Exception has occurred: {}".format(exp))
            self._has_exception = True
            self.done_training = True

    @timeit
    def test(self, test_x, remaining_time_budget=None):
        """
        :param test_x: list of vectors, input test models raw data.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        """
        if self.done_training is True or self._has_exception is True:
            return self._last_pred

        try:
            if self.train_loop_num == 1:
                self._context.raw_test_data = test_x

            preds = self._context.stage.test()

            if self._context.use_knn_fix and not (self._context.train_y_hat is None and self._context.train_y_true is None):
                log("Use knn fix test")
                preds = knn_test_fix(preds, self._context.train_y_hat, self._context.train_y_true)

            if self._context.is_finished:
                log("Finish all stages")
                self.done_training = True

            self._last_pred = preds
        except MemoryError as mem_error:
            log("MemoryError has occurred: {}".format(mem_error))
            self._has_exception = True
            self.done_training = True
        except Exception as exp:
            log("Exception has occurred: {}".format(exp))
            self._has_exception = True
            self.done_training = True

        return self._last_pred


if __name__ == '__main__':
    from ingestion.dataset import AutoSpeechDataset
    D = AutoSpeechDataset(os.path.join("../sample_data/DEMO", 'train.data'))
    D.read_dataset()
    m = Model(D.get_metadata())
    m.train(D.get_train())
    m.test(D.get_test())
    m.train(D.get_train())
    m.test(D.get_test())
    m.train(D.get_train())
    m.test(D.get_test())
    # m.train(D.get_train())
    # m.test(D.get_test())

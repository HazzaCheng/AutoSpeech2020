#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import time
from collections import OrderedDict
from typing import Any

from keras.callbacks import Callback
from keras.preprocessing import sequence
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import keras.backend as K
import tensorflow as tf
nesting_level = 0


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")


def timeit(method, start_log=None):
    def wrapper(*args, **kw):
        global nesting_level

        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        return result

    return wrapper

@timeit
def is_multilabel(labels):
    is_multilabel = False
    for label in labels:
        if sum(label) > 1:
            is_multilabel = True
            break
    return is_multilabel


def pad_seq(data, pad_len):
    return sequence.pad_sequences(data, maxlen=pad_len, dtype='float32', padding='post', truncating='post')


def ohe2cat(label):
    return np.argmax(label, axis=1)


def balanced_acc_for_keras(y_true, y_pred):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    C = tf.confusion_matrix(y_true, y_pred)
    C = tf.cast(C, dtype=np.float32)

    per_class = tf.diag_part(C) / tf.reduce_sum(C, 1)
    per_class = tf.gather_nd(per_class, tf.where(~tf.is_nan(per_class)))    # remove nan value
    score = K.mean(per_class)
    return score


def balanced_acc_metric(solution, prediction):
    solution = np.argmax(solution, axis=1)
    prediction = np.argmax(prediction, axis=1)
    C = confusion_matrix(solution, prediction)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score


def auc_metric(solution, prediction):
    solution = np.array(solution, dtype=np.float)
    prediction = np.array(prediction, dtype=np.float)
    if solution.sum(axis=0).min() == 0:
        return 0
    auc = roc_auc_score(solution, prediction, average='macro')

    return np.mean(auc * 2 - 1)


@timeit
def get_specified_length(x, ratio=0.95):
    """
    Get the specified length cover ratio% data, if ration = None, return mean length.
    """
    lens = [len(_) for _ in x]

    if ratio == None:
        specified_len = sum(lens) // len(lens)
    else:
        lens.sort()
        specified_len = lens[int(len(lens) * ratio)]

    return specified_len


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor="acc", baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print("Epoch %d: Reached baseline, terminating training" % (epoch))
                self.model.stop_training = True


def set_mp(processes=4):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except BaseException:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


class TimeBudget:
    def __init__(self, time_budget):
        self._time_budget = time_budget
        self._start_time = time.time()

    def reset(self):
        self._start_time = time.time()

    @property
    def remain(self):
        escape_time = time.time() - self._start_time
        return self._time_budget - escape_time

    @remain.setter
    def remain(self, value):
        self._time_budget = value

    def __add__(self, other):
        # self._time_budget += other
        return self

    def __sub__(self, other):
        # self._time_budget -= other
        return self

    def __str__(self):
        return str(self.remain)

    def __repr__(self):
        return repr(self.remain)

    def __format__(self, format_spec):
        return format(self.remain, format_spec)

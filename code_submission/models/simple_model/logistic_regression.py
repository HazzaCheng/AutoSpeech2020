#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/5 10:35
# @Author:  Mecthew

import numpy as np
from sklearn.linear_model import logistic
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from models.my_classifier import Classifier
from tools import timeit, ohe2cat, log


# Consider use LR as the first model because it can reach high point at
# first loop
class LogisticRegression(Classifier):
    def __init__(self):
        # TODO: init model, consider use CalibratedClassifierCV
        # TODO support multilabel
        log("new {}".format(self.__class__.__name__))

        self._num_classes = None
        self._is_multilabel = False

        self._model = None
        from models import LR_MODEL
        self.model_name = LR_MODEL
        self.is_init = False

    def init_model(self,
                   num_classes,
                   max_iter=200,
                   C=1.0,
                   is_multilabel=False,
                   **kwargs):
        self._num_classes = num_classes
        self._is_multilabel = is_multilabel
        if num_classes <= 5:
            class_weight = None
        else:
            class_weight = "balanced"

        self._model = logistic.LogisticRegression(
            C=C, max_iter=max_iter, solver='liblinear', multi_class='auto', class_weight=class_weight)

        if is_multilabel:
            self._model = OneVsRestClassifier(self._model)

        self.is_init = True

    def fit(self, x_train, y_train, *args, **kwargs):
        if not self._is_multilabel:
            self._model.fit(x_train, ohe2cat(y_train))
        else:
            self._model.fit(x_train, y_train)

    def predict(self, x_test, batch_size=32, *args, **kwargs):
        preds = self._model.predict_proba(x_test)

        return preds

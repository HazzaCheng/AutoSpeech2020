#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/27 10:12
# @Author:  Mecthew
import keras
import tensorflow as tf
from keras import optimizers
from keras.layers import (SpatialDropout1D, Input, Bidirectional, GlobalMaxPool1D,
                          Dense, Dropout, CuDNNLSTM, Activation)
from keras.models import Model as TFModel

from CONSTANT import VERBOSE
from models.neural_model.attention import Attention

from models.my_classifier import Classifier
from tools import log, ohe2cat, pad_seq
from data_generator import ModelSequenceDataGenerator


class BilstmAttentionModel(Classifier):
    def __init__(self):
        # clear_session()
        log("new {}".format(self.__class__.__name__))
        self._model = None
        self.is_init = False
        from models import BILSTM_MODEL
        self.model_name = BILSTM_MODEL
        self._is_multilabel = False
        self._class_num = None

    def preprocess_data(self, x, val_x=None, feature_length=None):
        x = pad_seq(x, pad_len=feature_length)

        if val_x is not None:
            val_x = pad_seq(val_x, pad_len=feature_length)
            return x, val_x

        return x

    def init_model(self,
                   input_shape,
                   num_classes,
                   is_multilabel,
                   **kwargs):
        self._class_num = num_classes
        self._is_multilabel = is_multilabel
        if num_classes == 2:
            loss = 'binary_crossentropy'
            output_activation = 'sigmoid'
            output_units = 1
        else:
            if self._is_multilabel:
                output_activation = "sigmoid"
            else:
                output_activation = 'softmax'
            loss = 'categorical_crossentropy'
            output_units = num_classes
        inputs = Input(shape=(None, input_shape[1]))
        lstm_1 = Bidirectional(CuDNNLSTM(64, name='blstm_1',
                                         return_sequences=True),
                               merge_mode='concat')(inputs)
        activation_1 = Activation('tanh')(lstm_1)
        dropout1 = SpatialDropout1D(0.5)(activation_1)
        attention_1 = Attention(8, 16)([dropout1, dropout1, dropout1])
        pool_1 = GlobalMaxPool1D()(attention_1)
        dropout2 = Dropout(rate=0.5)(pool_1)
        dense_1 = Dense(units=256, activation='relu')(dropout2)
        outputs = Dense(units=output_units, activation=output_activation)(dense_1)

        model = TFModel(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(
            # learning_rate=1e-3,
            lr=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=0.0002,
            amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit, params, epochs, **kwargs):
        patience = 2
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience)]
        val_x, val_y = validation_data_fit
        if self._class_num == 2:
            train_y = ohe2cat(train_y)
            val_y = ohe2cat(val_y)
        batch_size = params["batch_size"]
        steps_per_epoch = int(len(train_x) // batch_size)
        train_data_generator = ModelSequenceDataGenerator(train_x, train_y, **params)

        history = self._model.fit_generator(train_data_generator,
                                            steps_per_epoch=steps_per_epoch,
                                            validation_data=(val_x, val_y),
                                            epochs=epochs,
                                            max_queue_size=10,
                                            callbacks=callbacks,
                                            use_multiprocessing=False,
                                            workers=1,
                                            verbose=VERBOSE)
        return history

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)

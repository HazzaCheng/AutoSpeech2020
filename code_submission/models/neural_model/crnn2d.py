#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/26 21:29
# @Author:  Mecthew
import keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import (Input, Dense, Dropout, Convolution2D,
                          MaxPooling2D, ELU, Reshape, CuDNNGRU)
from keras.layers.normalization import BatchNormalization
from keras.models import Model as TFModel

from CONSTANT import VERBOSE
from models.my_classifier import Classifier
from tools import ohe2cat, log, pad_seq
from data_generator import ModelSequenceDataGenerator


class Crnn2dModel(Classifier):
    def __init__(self):
        # clear_session()
        log("new {}".format(self.__class__.__name__))
        self._model = None
        self.is_init = False
        from models import CRNN2D_MODEL
        self.model_name = CRNN2D_MODEL
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

        channel_axis = 3
        channel_size = 128
        min_size = min(input_shape[0], input_shape[1])
        feature_input = Input(shape=(None, input_shape[1], 1))

        # Conv block 1
        x = Convolution2D(64, 3, strides=(1, 1), padding='same', name='conv1')(feature_input)
        x = BatchNormalization(axis=channel_axis, name='bn1')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
        x = Dropout(0.1, name='dropout1')(x)

        # Conv block 2
        x = Convolution2D(channel_size, 3, strides=(1, 1), padding='same', name='conv2')(x)
        x = BatchNormalization(axis=channel_axis, name='bn2')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
        x = Dropout(0.1, name='dropout2')(x)

        # Conv block 3
        x = Convolution2D(channel_size, 3, strides=(1, 1), padding='same', name='conv3')(x)
        x = BatchNormalization(axis=channel_axis, name='bn3')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
        x = Dropout(0.1, name='dropout3')(x)

        if min_size // 24 >= 4:
            # Conv block 4
            x = Convolution2D(
                channel_size,
                3,
                strides=(1, 1),
                padding='same',
                name='conv4')(x)
            x = BatchNormalization(axis=channel_axis, name='bn4')(x)
            x = ELU()(x)
            x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
            x = Dropout(0.1, name='dropout4')(x)

        x = Reshape((-1, channel_size))(x)

        # gru_units = max(int(num_classes*1.5), 128)
        gru_units = 256
        # GRU block 1, 2, output
        x = CuDNNGRU(gru_units, return_sequences=True, name='gru1')(x)
        x = CuDNNGRU(gru_units, return_sequences=False, name='gru2')(x)
        # x = Dense(max(int(num_classes*1.5), 128), activation='relu', name='dense1')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(output_units, activation=output_activation, name='output')(x)

        model = TFModel(inputs=feature_input, outputs=outputs)
        optimizer = optimizers.Adam(
            # learning_rate=1e-3,
            lr=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=1e-4,
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
        train_x_4dims = train_x[:, :, :, np.newaxis]
        val_x, val_y = validation_data_fit
        val_x_4dims = val_x[:, :, :, np.newaxis]
        batch_size = params["batch_size"]
        steps_per_epoch = len(train_x) // batch_size
        if self._class_num == 2:
            train_y = ohe2cat(train_y)
            val_y = ohe2cat(val_y)
        train_data_generator = ModelSequenceDataGenerator(train_x_4dims, train_y, **params)

        history = self._model.fit_generator(train_data_generator,
                                            validation_data=(val_x_4dims, val_y),
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=epochs,
                                            max_queue_size=10,
                                            callbacks=callbacks,
                                            use_multiprocessing=False,
                                            workers=1,
                                            verbose=VERBOSE)
        return history

    def predict(self, x_test, batch_size=32):
        x_test = np.expand_dims(x_test, axis=-1)
        return self._model.predict(x_test, batch_size=batch_size)

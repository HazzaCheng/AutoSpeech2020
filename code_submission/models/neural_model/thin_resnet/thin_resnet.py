#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-03-23
import os
import time

from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback

from CONSTANT import VERBOSE
from models.my_classifier import Classifier
from models.neural_model.thin_resnet.backbone import choose_net
from models.neural_model.thin_resnet.data_generator import DataGenerator
from tools import log
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np

WEIGHT_DECAY = 1e-3


class ThinResnet(Classifier):
    def __init__(self):
        # clear_session()
        log("new {}".format(self.__class__.__name__))
        self._model = None
        self._pretrain_model_path = os.path.join(os.path.dirname(__file__), 'pretrained_models/thin_resnet34.h5')
        self._num_class_threshold = 37
        self._frozen_layer_num = 124
        # self._frozen_layer_num_mild = 100 # qmc's inspirion
        self._frozen_layer_num_mild = 124

        self._callbacks = []

        self.is_init = False
        self.use_dropout = False
        from models import THINRESNET_MODEL
        self.model_name = THINRESNET_MODEL

    def init_model(self,
                   input_shape,
                   num_classes,
                   is_multilabel,
                   train_num,
                   vlad_clusters=8,
                   ghost_clusters=2,
                   bottleneck_dim=512,
                   aggregation='gvlad',
                   mode='train',
                   **kwargs):
        if is_multilabel:
            loss = 'sigmoid'
        else:
            loss = 'softmax'

        backbone_net = choose_net('thin_resnet')
        inputs, x = backbone_net(input_shape=input_shape, mode=mode)
        x_fc = keras.layers.Conv2D(bottleneck_dim, (7, 1),
                                   strides=(1, 1),
                                   activation='relu',
                                   # padding='same',
                                   kernel_initializer='orthogonal',
                                   use_bias=True, trainable=True,
                                   kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                   bias_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                   name='x_fc')(x)

        if aggregation == 'avg':
            if mode == 'train':
                x = keras.layers.AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
                x = keras.layers.Reshape((-1, bottleneck_dim))(x)
            else:
                x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
                x = keras.layers.Reshape((1, bottleneck_dim))(x)
        elif aggregation == 'vlad':
            x_k_center = keras.layers.Conv2D(vlad_clusters, (7, 1),
                                             strides=(1, 1),
                                             kernel_initializer='orthogonal',
                                             use_bias=True, trainable=True,
                                             kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                             bias_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                             name='vlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x_fc, x_k_center])
        elif aggregation == 'gvlad':
            x_k_center = keras.layers.Conv2D(vlad_clusters + ghost_clusters, (7, 1),
                                             strides=(1, 1),
                                             # padding='same',
                                             kernel_initializer='orthogonal',
                                             use_bias=True,
                                             trainable=True,
                                             kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                             bias_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                             name='gvlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')(
                [x_fc, x_k_center])
        else:
            raise IOError('==> unknown aggregation mode')

        x = keras.layers.Dense(bottleneck_dim, activation='relu',
                               kernel_initializer='orthogonal',
                               use_bias=True, trainable=True,
                               kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                               bias_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                               name='fc6')(x)

        if loss == 'softmax':
            y = keras.layers.Dense(num_classes, activation='softmax',
                                   kernel_initializer='orthogonal',
                                   use_bias=False, trainable=True,
                                   kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                   bias_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                   name='prediction')(x)
            trnloss = 'categorical_crossentropy'

        elif loss == 'amsoftmax':
            x_l2 = keras.layers.Lambda(lambda x: K.l2_normalize(x, 1))(x)
            y = keras.layers.Dense(num_classes,
                                   kernel_initializer='orthogonal',
                                   use_bias=False, trainable=True,
                                   kernel_constraint=keras.constraints.unit_norm(),
                                   kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                   bias_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                   name='prediction')(x_l2)
            trnloss = amsoftmax_loss
        elif loss == 'sigmoid':
            y = keras.layers.Dense(num_classes, activation='sigmoid',
                                   kernel_initializer='orthogonal',
                                   use_bias=False, trainable=True,
                                   kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                   bias_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                   name='prediction')(x)
            trnloss = 'binary_crossentropy'
        else:
            raise IOError('unknown loss')

        if mode == 'pretrain':
            model = keras.models.Model(inputs, x, name='vggvox_resnet2D_{}_{}'.format(loss, aggregation))
        elif mode == 'pred':
            model = keras.models.Model(inputs, y, name='vggvox_resnet2D_{}_{}'.format(loss, aggregation))
        else:
            raise IOError('unsupported mode')

        if os.path.isfile(self._pretrain_model_path):
            log(f"{self.model_name} read pretrain model")
            model.load_weights(self._pretrain_model_path, by_name=True, skip_mismatch=True)
            if num_classes >= 37:
                frz_layer_num = self._frozen_layer_num
            else:
                frz_layer_num = self._frozen_layer_num_mild
            for layer in model.layers[: frz_layer_num]:
                layer.trainable = False

        pretrain_output = model.output
        weight_decay = WEIGHT_DECAY
        if is_multilabel:
            output_activation = "sigmoid"
            loss_func = "binary_crossentropy"
        else:
            output_activation = "softmax"
            loss_func = "categorical_crossentropy"
        # self.use_dropout = train_num <= 300
        # if self.use_dropout and False:
        #     pretrain_output = keras.layers.Dropout(rate=0.2)(pretrain_output)
        y = keras.layers.Dense(
            num_classes,
            activation=output_activation,
            kernel_initializer="orthogonal",
            use_bias=False,
            trainable=True,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name="prediction",
        )(pretrain_output)
        model = keras.models.Model(model.input, y, name="vggvox_resnet2D_{}_{}_new".format(output_activation, "gvlad"))
        opt = keras.optimizers.Adam(lr=1e-3)
        model.compile(optimizer=opt, loss=loss_func, metrics=["acc"])

        model.summary()

        self._model = model
        self.is_init = True

        self._callbacks.append(EarlyStopping(monitor="val_loss", patience=15))
        self._callbacks.append(LearningRateScheduler(self.step_decay))

    def step_decay(self, epoch):
        # TODO just fixed lr
        lr = 0.00175

        return np.float(lr)

    def fit(self, train_x, train_y, validation_data_fit, epochs, cur_loop_num, params, **kwargs):
        val_x, val_y = validation_data_fit

        # if self._is_multilabel:
        #     train_y = train_y
        #     val_y = val_y
        # else:
        #     train_y = ohe2cat(train_y)
        #     val_y = ohe2cat(val_y)

        callbacks = self._callbacks + params["callbacks"]
        batch_size = params["batch_size"]
        if cur_loop_num <= 1:
            steps_per_epoch = int(len(train_x) // batch_size // 2)
        else:
            steps_per_epoch = int(len(train_x) // batch_size)

        train_data_generator = DataGenerator(train_x, train_y, **params)

        s = time.time()
        history = self._model.fit_generator(train_data_generator,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=epochs,
                                            max_queue_size=10,
                                            callbacks=callbacks,
                                            use_multiprocessing=False,
                                            workers=1,
                                            verbose=VERBOSE)
        t = time.time()
        log(f"$$$ train time {t - s}")

        return history

    def predict(self, x_test, batch_size=8):
        s = time.time()
        K.set_learning_phase(0)
        preds = self._model.predict(x_test, batch_size=batch_size)
        t = time.time()
        log(f"$$$ test time {t - s}")
        return preds


class VladPooling(keras.engine.Layer):
    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers + self.g_centers, input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers * input_shape[0][-1])

    def call(self, x):
        feat, cluster_score = x
        num_features = feat.shape[-1]
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims=True)
        A = K.expand_dims(A, -1)  # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(feat, -2)  # feat_broadcast : bz x W x H x 1 x D
        feat_res = feat_broadcast - self.cluster  # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(A, feat_res)  # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

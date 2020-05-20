#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-03-19
import time

import gc
from sklearn.preprocessing import StandardScaler

from CONSTANT import SPEECH_SIMPLE_MODEL_MAX_SAMPLE_NUM, SPEECH_SIMPLE_MODEL_EACH_LOOP_SAMPLE_NUM, \
    SPEECH_NEURAL_MODEL_EACH_LOOP_SAMPLE_NUM, FE_RS_SPEC_LEN_CONFIG_AGGR, FE_RS_SPEC_LEN_CONFIG_MILD, SR, \
    FE_RS_SPEC_LEN_CONFIG_LONG, EXPLORATION_STAGE_SPEC_LEN
from features import MAX_FRAME_NUM, AUDIO_SAMPLE_RATE, MFCC, get_specified_feature_func, get_features, NCPU, \
    STFT, MEL_SPECTROGRAM, pad_data_by_copy
from models import LR_MODEL, LSTM_MODEL, CRNN2D_MODEL, BILSTM_MODEL, ATT_GRU_MODEL, SPEECH_MODEL_LIB, CNN2D_MODEL, \
    THINRESNET_MODEL
from KapreFeatsMaker import KapreMelSpectroGramFeatsMaker
from stage import Stage, Preds
from tools import log, timeit, auc_metric, get_specified_length, pad_seq, TerminateOnBaseline, set_mp, \
    balanced_acc_for_keras, balanced_acc_metric
import numpy as np
from keras import backend as K


class FastStage(Stage):

    @timeit
    def __init__(self, context, end_loop_num, fea_name, pre_func, max_duration, **kwargs):
        super(FastStage, self).__init__(context, end_loop_num, fea_name, pre_func, **kwargs)
        self._kwargs = kwargs
        self._raw_data_cut_length = min(max_duration * AUDIO_SAMPLE_RATE, self.ctx.cut_length)
        log("ctx_cut_length: {}".format(self.ctx.cut_length))
        log("raw_data_cut_length: {}".format(self._raw_data_cut_length))

        self._fea_name = self._fea_name + '_' + str(self._raw_data_cut_length / AUDIO_SAMPLE_RATE) + 's'
        self._stage_name = 'FastStage'

        self._model = SPEECH_MODEL_LIB[LR_MODEL]()

        self._use_kapre_feat_maker = True
        if self._use_kapre_feat_maker:
            self._kapre_melfeat_maker = KapreMelSpectroGramFeatsMaker(self._raw_data_cut_length)

        kwargs = {
            'num_classes': self.ctx.num_classes,
            'max_iter': 100,
            'is_multilabel': self.ctx.is_multilabel,
        }
        self._model.init_model(**kwargs)
        log(f"{self._stage_name} init; feature {self._fea_name}")

    @timeit
    def _preprocess_data(self):
        super(FastStage, self)._preprocess_data()
        if self._stage_loop_num == 1:
            sample_num = SPEECH_SIMPLE_MODEL_EACH_LOOP_SAMPLE_NUM * self._stage_loop_num
        elif self._stage_loop_num < self._stage_end_loop_num:
            sample_num = SPEECH_SIMPLE_MODEL_EACH_LOOP_SAMPLE_NUM * self._stage_loop_num
        elif self._stage_loop_num == self._stage_end_loop_num:
            sample_num = SPEECH_SIMPLE_MODEL_MAX_SAMPLE_NUM
        else:
            return

        if self._stage_loop_num != self._stage_end_loop_num:
            sample_index = self.ctx.get_samples(
                per_class_sample_num=3 * self._stage_loop_num,
                min_sample_num=sample_num,
                max_sample_num=0,
                use_all_data=False,
                replace=False,
                reset=False)
        else:
            self._pre_train_x = None
            self._pre_train_y = None
            sample_index = self.ctx.get_samples(
                per_class_sample_num=10,
                min_sample_num=2000,
                max_sample_num=2000,
                use_all_data=True,
                replace=True,
                reset=True)

        if self._stage_loop_num == self._stage_end_loop_num - 1:
            # TODO donot calculate auc
            val_x, val_y = self.ctx.val_data
            self._pre_val_x = [x[:self._raw_data_cut_length] for x in val_x]
            if self._use_kapre_feat_maker:
                self._pre_val_x = self._kapre_melfeat_maker.make_features(self._pre_val_x)
            else:
                self._pre_val_x = self._pre_func(self._pre_val_x)
                self._pre_val_x = self._normalize_data(self._pre_val_x)
            self._pre_val_y = val_y
            # pass

        need_pre = list(set([i for i in sample_index if i not in self._pre_train_x_dict]))
        log(f"{self._stage_loop_num} use {len(sample_index)} new train samples,"
            f" preprocess {len(need_pre)} train samples")
        if len(need_pre) <= 0:
            return

        # train_x, train_y = self.ctx.train_data
        train_x, train_y = self.ctx.raw_data
        raw_data = [train_x[i][:self._raw_data_cut_length] for i in need_pre]

        if self._use_kapre_feat_maker:
            pre_data = self._kapre_melfeat_maker.make_features(raw_data)
        else:
            pre_data = self._pre_func(raw_data)
            pre_data = self._normalize_data(pre_data)

        for i, sample_id in enumerate(need_pre):
            self._pre_train_x_dict[sample_id] = pre_data[i]

        x = [self._pre_train_x_dict[i] for i in sample_index]
        y = [train_y[i] for i in sample_index]

        if self._pre_train_x is None:
            self._pre_train_x = np.asarray(x, dtype=np.float32)
            self._pre_train_y = np.asarray(y, dtype=np.int32)
        else:
            self._pre_train_x = np.concatenate((self._pre_train_x, x), axis=0)
            self._pre_train_y = np.concatenate((self._pre_train_y, y), axis=0)

    @timeit
    def _normalize_data(self, data):
        x_feas = []
        for x in data:
            fea = np.mean(x, axis=0).reshape(-1)
            fea_std = np.std(x, axis=0).reshape(-1)
            x_feas.append(np.concatenate([fea, fea_std], axis=-1))
        x_feas = np.asarray(x_feas, dtype=np.float32)
        scaler = StandardScaler()
        pre_data = scaler.fit_transform(x_feas[:, :])
        return pre_data

    def _transition(self):
        del self._model
        self._model = None
        if self._use_kapre_feat_maker:
            del self._kapre_melfeat_maker
        gc.collect()

        # fea_name = MFCC
        # max_duration = 10
        # end_loop_num = 20
        # fea_pre_func = get_specified_feature_func(fea_name, n_mfcc=96)
        # self.ctx.stage = ExplorationStage(self._context, end_loop_num, fea_name,
        #                                   fea_pre_func, max_duration)

        # fea_name = STFT
        # end_loop_num = 2000
        # self.ctx.stage = EnhancementStage(self._context, end_loop_num, fea_name, None)
        self.ctx.stage = self.ctx.resnet_stage
        log(f"change {self._stage_name} to {self.ctx.stage._stage_name}")

    @timeit
    def train(self, remain_time_budget=None):
        super(FastStage, self).train(remain_time_budget)
        self.ctx.is_first_train = False
        self._preprocess_data()

        log(f"train_x shape {self._pre_train_x.shape} train_y shape {self._pre_train_y.shape}")
        self._model.fit(self._pre_train_x, self._pre_train_y)
        if self.ctx.use_knn_fix:
            self.ctx.train_y_true = self._pre_train_y
            self.ctx.train_y_hat = self._model.predict(self._pre_train_x)

    @timeit
    def test(self, remain_time_budget=None):
        super(FastStage, self).test(remain_time_budget)

        score = 0
        if self._stage_loop_num == self._stage_end_loop_num - 1:
            # TODO donot calculate auc
            # score = auc_metric(self._pre_val_y, self._model.predict(self._pre_val_x))
            score = balanced_acc_metric(self._pre_val_y, self._model.predict(self._pre_val_x))
        if self._stage_loop_num == self._stage_end_loop_num:
            self._need_transition = True

        if self._pre_test_x is None:
            test_x = self.ctx.raw_test_data
            self._pre_test_x = [x[:self._raw_data_cut_length] for x in test_x]

            if self._use_kapre_feat_maker:
                self._pre_test_x = self._kapre_melfeat_maker.make_features(self._pre_test_x)
            else:
                self._pre_test_x = self._pre_func(self._pre_test_x)
                self._pre_test_x = self._normalize_data(self._pre_test_x)

        log(f"test_x shape {self._pre_test_x.shape}")
        preds = self._model.predict(self._pre_test_x)
        self._stage_test_loop_num += 1
        # self.ctx.add_predicts(Preds(self._model.model_name, self._fea_name, score * 0.8, preds))
        log("lr predicts, val_score {}".format(score))
        self.ctx.add_predicts(Preds(self._model.model_name, self._fea_name, score, preds))

        if self._need_transition:
            self.ctx.lr_last_preds = preds
            self._transition()
        # TODO whether ensemble
        # return self.ctx.ensemble_predicts()
        return preds


class ExplorationStage(Stage):

    @timeit
    def __init__(self, context, end_loop_num, fea_name, pre_func, max_duration, **kwargs):
        super(ExplorationStage, self).__init__(context, end_loop_num, fea_name, pre_func, **kwargs)
        self._kwargs = kwargs
        # self._raw_data_cut_length = min(max_duration * AUDIO_SAMPLE_RATE, self.ctx.cut_length)
        self._raw_data_cut_length = self.ctx.cut_length
        self._fea_name = self._fea_name + '_' + str(self._raw_data_cut_length // AUDIO_SAMPLE_RATE) + 's'
        self._stage_name = 'ExplorationStage'

        self._input_shape = None
        self._fea_length = None

        self._model = None
        self._models = [LSTM_MODEL, CRNN2D_MODEL, BILSTM_MODEL]
        self._existing_models_dict = {}
        self._model_idx = 0
        self._first_model_sample_loop = 6
        self._model_loop = 0
        self._model_max_loop = 15
        self._round_num = 0

        self._cur_model_max_score = 0
        self._not_rise_time = 0
        self._rise_time = 0
        self._patience = 3

        self._pre_val_x_unfixed = None
        self._pre_test_x_unfixed = None

        self._spec_len_status = 0
        self._feature_params = {
            "mp_pooler": set_mp(processes=NCPU),
            "train_spec_len": None,
            "test_spec_len": None,
            "mode": "train",
            "feature": fea_name,
            "n_classes": self.ctx.num_classes,
            "callbacks": [],
            "batch_size": 32,
        }

        log(f"{self._stage_name} init; feature {self._fea_name}")

    @timeit
    def _preprocess_data(self):
        super(ExplorationStage, self)._preprocess_data()
        # if self._stage_loop_num < self._first_model_sample_loop:
        #     sample_num = SPEECH_NEURAL_MODEL_EACH_LOOP_SAMPLE_NUM * self._stage_loop_num
        # elif self._stage_loop_num == self._first_model_sample_loop:
        #     sample_num = self.ctx.train_num
        # else:
        #     return

        sample_index = self.ctx.get_samples(
            per_class_sample_num=10,
            min_sample_num=300,
            max_sample_num=max(500, self.ctx.num_classes * 3),
            replace=True, reset=False)
        need_pre = list(set([i for i in sample_index if i not in self._pre_train_x_dict]))
        log(f"{self._stage_loop_num} use {len(sample_index)} train samples, preprocess {len(need_pre)} train samples")
        # train_x, train_y = self.ctx.train_data
        train_x, train_y = self.ctx.raw_data
        if len(need_pre) > 0:
            raw_data = [train_x[i][:self._raw_data_cut_length] for i in need_pre]
            pre_data = self._pre_func(raw_data)
            if self._fea_length is None:
                self._fea_length = get_specified_length(pre_data)
                self._feature_length = min(MAX_FRAME_NUM, self._fea_length)
                log(f"{self._stage_name} feature length {self._fea_length}")
            # pre_data = self._format_data(pre_data, mode="train")
            for i, sample_id in enumerate(need_pre):
                self._pre_train_x_dict[sample_id] = pre_data[i]

        x = [self._pre_train_x_dict[i] for i in sample_index]
        x = self._format_data(x, mode="train")
        y = [train_y[i] for i in sample_index]
        self._pre_train_x = np.asarray(x, dtype=np.float32)
        self._pre_train_y = np.asarray(y, dtype=np.int32)

        # if self._pre_val_x is None or self._spec_len_status == 1:
        if self._pre_val_x_unfixed is None:
            val_x, val_y = self.ctx.val_data
            self._pre_val_x_unfixed = [x[:self._raw_data_cut_length] for x in val_x]
            self._pre_val_x_unfixed = self._pre_func(self._pre_val_x_unfixed)
        if self._spec_len_status == 1:
            self._pre_val_x = self._format_data(self._pre_val_x_unfixed, mode="test")
            self._pre_val_y = self.ctx.val_data[1]

        if self._input_shape is None:
            self._input_shape = self._pre_train_x.shape

    def _update_feature_params(self):
        spec_lens = EXPLORATION_STAGE_SPEC_LEN[self._round_num]
        self._feature_params["train_spec_len"] = spec_lens[0]
        self._feature_params["test_spec_len"] = spec_lens[1]
        if self._model_loop == 1:
            self._spec_len_status = 1

    @timeit
    def _format_data(self, data, mode):
        # pre_data = pad_seq(data, pad_len=self._fea_length)
        mp_pooler = self._feature_params["mp_pooler"]
        if mode == "train":
            spec_len = self._feature_params["train_spec_len"]
            x = [mp_pooler.apply_async(pad_data_by_copy, args=(x, spec_len, mode))
                 for x in data]
            pre_data = np.array([p.get() for p in x])
        else:
            spec_len = self._feature_params["test_spec_len"]
            x = [mp_pooler.apply_async(pad_data_by_copy, args=(x, spec_len, mode))
                 for x in data]
            pre_data = np.array([p.get() for p in x])

        return pre_data

    def _transition(self):
        # del self._model
        # self._model = None
        # K.clear_session()
        # gc.collect()
        for model_name in self._existing_models_dict.keys():
            sequence_model = self._existing_models_dict[model_name]
            del sequence_model
            K.clear_session()
            gc.collect()

        # fea_name = STFT
        # end_loop_num = 200
        # self.ctx.stage = EnhancementStage(self._context, end_loop_num, fea_name, None)

        self.ctx.is_last_stage = True
        if self.ctx.is_last_stage:
            self.ctx.is_finished = True
        log(f"finish stage {self._stage_name}")

    def _update_model(self):
        if self._model is None or \
                self._not_rise_time == self._patience or \
                self._model_loop == self._model_max_loop:
            if self._model_idx == len(self._models):
                self._round_num += 1
                self._model_idx = 0
                if self._round_num >= len(EXPLORATION_STAGE_SPEC_LEN):
                    self._need_transition = True
                    return
            model_name = self._models[self._model_idx]
            log(f"start using no.{self._model_idx} {model_name}")

            # empty memory
            # del self._model
            # self._model = None
            # K.clear_session()
            # gc.collect()

            if model_name == CNN2D_MODEL:
                kwargs = {
                    'input_shape': self._input_shape[1:],
                    'num_classes': self.ctx.num_classes,
                    'max_layer_num': 10
                }
            elif model_name in [LSTM_MODEL, BILSTM_MODEL, CRNN2D_MODEL, ATT_GRU_MODEL]:
                kwargs = {
                    'input_shape': self._input_shape[1:],
                    'num_classes': self.ctx.num_classes,
                    'is_multilabel': self.ctx.is_multilabel
                }
            else:
                raise Exception("No such model config!")

            if model_name in self._existing_models_dict.keys():
                self._model = self._existing_models_dict[model_name]
            else:
                self._model = SPEECH_MODEL_LIB[model_name]()
                self._model.init_model(**kwargs)
                self._existing_models_dict[model_name] = self._model

            self._cur_model_max_score = 0
            self._not_rise_time = 0
            self._rise_time = 0
            self._model_loop = 0
            self._model_idx += 1

    @timeit
    def train(self, remain_time_budget=None):
        super(ExplorationStage, self).train(remain_time_budget)
        self._model_loop += 1
        self._update_feature_params()
        self._preprocess_data()

        if self._model is None:
            self._update_model()

        log("model {} begin trainning".format(self._model.model_name))
        log(f"train_x shape {self._pre_train_x.shape} train_y shape {self._pre_train_y.shape}")
        self._model.fit(self._pre_train_x, self._pre_train_y,
                        validation_data_fit=(self._pre_val_x, self._pre_val_y),
                        params=self._feature_params,
                        epochs=10)
        if self.ctx.use_knn_fix:
            self.ctx.train_y_true = self._pre_train_y
            self.ctx.train_y_hat = self._model.predict(self._pre_train_x)

    @timeit
    def test(self, remain_time_budget=None):
        super(ExplorationStage, self).test(remain_time_budget)

        # score = auc_metric(self._pre_val_y, self._model.predict(self._pre_val_x))
        score = balanced_acc_metric(self._pre_val_y, self._model.predict(self._pre_val_x))
        if score < self._cur_model_max_score:
            self._not_rise_time += 1
            self._rise_time = 0
            log(
                f"{self._model.model_name} {score} not rise for {self._not_rise_time} times, current max score {self._cur_model_max_score}")
        else:
            self._cur_model_max_score = score
            self._not_rise_time = 0
            self._rise_time += 1
            log(f"{self._model.model_name} {score} rise for {self._rise_time} times")

        if self._pre_test_x_unfixed is None:
            test_x = self.ctx.raw_test_data
            self._pre_test_x_unfixed = [x[:self._raw_data_cut_length] for x in test_x]
            self._pre_test_x_unfixed = self._pre_func(self._pre_test_x_unfixed)
        if self._pre_test_x is None or self._spec_len_status == 1:
            self._pre_test_x = self._format_data(self._pre_test_x_unfixed, mode="test")
            self._spec_len_status = 0

        if score < self.ctx.max_score:
            log(f"{score} not best score ({self.ctx.max_score}), use ensemble preds")
            if score > self.ctx.get_min_score_in_kbest():
                log("new predicts")
                cur_preds = self._model.predict(self._pre_test_x)
                self._stage_test_loop_num += 1
                self.ctx.add_predicts(Preds(self._model.model_name, self._fea_name, score, cur_preds))
            preds = self.ctx.ensemble_predicts()
        else:
            log(f"test_x shape {self._pre_test_x.shape}")
            preds = self._model.predict(self._pre_test_x)
            self._stage_test_loop_num += 1
            log("new predicts")
            self.ctx.add_predicts(Preds(self._model.model_name, self._fea_name, score, preds))

        self._update_model()
        if self._need_transition:
            self._transition()

        return preds


class EnhancementStage(Stage):

    @timeit
    def __init__(self, context, end_loop_num, fea_name, pre_func, **kwargs):
        super(EnhancementStage, self).__init__(context, end_loop_num, fea_name, pre_func, **kwargs)
        self._kwargs = kwargs
        self._fea_name = STFT
        self._stage_name = 'EnhancementStage'
        self._better_score_cnt = 0

        self._feature_params = {
            "train_spec_len": None,
            "test_spec_len": None,
            "train_wav_len": None,
            "test_wav_len": None,
            "spec_len": 250,

            "feature": self._fea_name,
            # "feature": STFT,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "sr": SR * 100,
            "n_mfcc": 96,
            "n_mels": 96,
        }
        if self._fea_name == STFT:
            self._input_shape = (257, 250, 1)
        elif self._fea_name == MFCC:
            self._input_shape = (self._feature_params['n_mfcc'], self._feature_params['spec_len'], 1)
        elif self._fea_name == MEL_SPECTROGRAM:
            self._input_shape = (self._feature_params['n_mels'], self._feature_params['spec_len'], 1)
        else:
            raise Exception("no such feature")

        self._model = SPEECH_MODEL_LIB[THINRESNET_MODEL]()
        kwargs = {
            'input_shape': self._input_shape,
            'num_classes': self.ctx.num_classes,
            'is_multilabel': self.ctx.is_multilabel,
            'mode': 'pretrain',
            "train_num": self.ctx._metadata["train_num"]
        }
        self._model.init_model(**kwargs)

        if self.ctx.num_classes >= 37:
            self._loop_spec_len = FE_RS_SPEC_LEN_CONFIG_AGGR
        else:
            self._loop_spec_len = FE_RS_SPEC_LEN_CONFIG_MILD

        self._spec_len_status = 0

        self._last_pred_loop = 0
        self._last_pred = None
        self._all_preds = {}

        self._loss = []
        self._acc = []
        self._train_history = {}
        self._prev50_topk_preds_ids = []
        self._prev50_stage_loop_num = None

        self._train_params = {
            "mp_pooler": set_mp(processes=NCPU),
            "dim": self._input_shape,
            "spec_len": self._feature_params['spec_len'],
            "mode": "train",
            "n_classes": self.ctx.num_classes,
            "callbacks": [],
            "batch_size": 32,
            "shuffle": True,
            "normalize": True,
        }

        # self._cur_model_max_score = 0
        # self._not_rise_time = 0
        # self._rise_time = 0
        # self._patience = 3

        log(f"{self._stage_name} init; feature {self._fea_name}")

    def _transition(self):
        del self._model
        self._model = None
        K.clear_session()
        gc.collect()

        if self._decide_use_all_data():
            self.ctx.is_finished = True
            log(f"finish stage {self._stage_name}")
        else:
            fea_name = MFCC
            max_duration = 20
            end_loop_num = 50
            fea_pre_func = get_specified_feature_func(fea_name, n_mfcc=96)
            self.ctx.stage = ExplorationStage(self._context, end_loop_num,
                                              fea_name, fea_pre_func,
                                              max_duration)
            self.ctx.is_last_stage = True

    @timeit
    def _preprocess_data(self):
        super(EnhancementStage, self)._preprocess_data()
        # TODO don't use val data ?
        x, y = self.ctx.raw_data
        use_all_data = self._decide_use_all_data()
        self._decide_win_length_and_hop_length()
        self._try_to_update_spec_len()

        # TODO all use deepwisdom parameters
        if self._stage_loop_num == 1:
            if self.ctx.cut_length >= 480000:
                log("use LONG SPEC LEN, cut_length: {}".format(self.ctx.cut_length))
                self._loop_spec_len = FE_RS_SPEC_LEN_CONFIG_LONG
                spec_len = 500
                self._feature_params['spec_len'] = spec_len
                self._train_params['spec_len'] = spec_len
                self._feature_params["train_spec_len"] = spec_len
                self._feature_params["test_spec_len"] = spec_len
                self._feature_params["train_wav_len"] = spec_len * self._feature_params["hop_length"]
                self._feature_params["test_wav_len"] = spec_len * self._feature_params["hop_length"]

            sample_index = self.ctx.get_samples(per_class_sample_num=10,
                                                min_sample_num=200,
                                                max_sample_num=max(300, int(self.ctx.num_classes * 2)),
                                                # max_sample_num=200,
                                                replace=True,
                                                reset=False,
                                                use_all_data=use_all_data)
        elif self._stage_loop_num >= 40:
            sample_index = self.ctx.get_samples(per_class_sample_num=10,
                                                min_sample_num=300,
                                                max_sample_num=max(300, int(self.ctx.num_classes * 2)),
                                                # max_sample_num=200,
                                                replace=True,
                                                reset=False,
                                                use_all_data=use_all_data)

        else:
            sample_index = self.ctx.get_samples(per_class_sample_num=10,
                                                min_sample_num=300,
                                                max_sample_num=max(500, self.ctx.num_classes * 3),
                                                # max_sample_num=300,
                                                replace=True,
                                                reset=False,
                                                use_all_data=use_all_data)

        if self._stage_loop_num in self._loop_spec_len:
            print(f"{self._stage_name} update spec len {self._feature_params}")

        if self._spec_len_status == 1:
            self._pre_train_x_dict.clear()
            self._spec_len_status = 2

        need_pre = list(set([i for i in sample_index if i not in self._pre_train_x_dict]))
        log(f"{self._stage_loop_num} use {len(sample_index)} train samples, preprocess {len(need_pre)} train samples")
        if len(need_pre) > 0:
            self._feature_params["mode"] = "train"
            raw_data = [x[i] for i in need_pre]
            pre_data = get_features(raw_data, self._feature_params)

            for i, sample_id in enumerate(need_pre):
                self._pre_train_x_dict[sample_id] = pre_data[i]

        x = [self._pre_train_x_dict[i] for i in sample_index]
        y = [y[i] for i in sample_index]

        # self._pre_train_x = np.asarray(x, dtype=np.float32)
        self._pre_train_x = x
        # self._pre_train_y = y
        self._pre_train_y = np.asarray(y, dtype=np.int32)

        if self._pre_val_x is None:
            val_x, val_y = self.ctx.val_data
            val_args = self._feature_params.copy()
            val_args["mode"] = "test"
            x = get_features(val_x, val_args)
            x = np.array(x, dtype=np.float32)
            self._pre_val_x = x[:, :, :, np.newaxis]
            self._pre_val_y = np.asarray(val_y, dtype=np.int32)

    def _try_to_update_spec_len(self):
        if self._stage_loop_num in self._loop_spec_len:
            train_spec_len, test_spec_len = self._loop_spec_len[self._stage_loop_num]
            self._update_spec_len(train_spec_len, test_spec_len)

    def _update_spec_len(self, train_spec_len, test_spec_len):
        self._feature_params["train_spec_len"] = train_spec_len
        self._feature_params["test_spec_len"] = test_spec_len
        self._feature_params["train_wav_len"] = train_spec_len * self._feature_params["hop_length"]
        self._feature_params["test_wav_len"] = test_spec_len * self._feature_params["hop_length"]
        self._feature_params["mode"] = "train"
        self._spec_len_status = 1
        # print(f"{self._stage_name} update spec len {self._feature_params}")
        return True

    def _decide_use_all_data(self):
        return self._better_score_cnt >= 1

    def _decide_win_length_and_hop_length(self):
        if self.ctx.mean_length <= 10 * AUDIO_SAMPLE_RATE:
            self._feature_params["win_length"] = 400
            self._feature_params["hop_length"] = 160
        elif self.ctx.mean_length <= 20 * AUDIO_SAMPLE_RATE:
            self._feature_params["win_length"] = 400
            self._feature_params["hop_length"] = 250
        else:
            self._feature_params["win_length"] = 400
            self._feature_params["hop_length"] = 340

    def _decide_epoch_num(self):
        if self._stage_loop_num == 1:
            if self.ctx.num_classes <= 10:
                epoch_num = 8
            else:
                epoch_num = 14
        else:
            epoch_num = 1
        return epoch_num

    def _decide_warmup_loops(self):
        if self.ctx.num_classes <= 10:
            return 2
        elif 10 < self.ctx.num_classes <= 37:
            return 8
        else:
            return 11

    def _is_predict(self):
        flag = (
                (self._stage_loop_num in self._loop_spec_len)
                or (self._stage_loop_num < 10)
                or (self._stage_loop_num < 21 and self._stage_loop_num % 2 == 1)
                or (self._stage_loop_num - self._last_pred_loop > 3)
                or self._stage_test_loop_num == 0
        )
        return flag

    def _is_ensenmble(self):
        ensemble_start_loop = 15
        if self._stage_loop_num > ensemble_start_loop:
            return True
        else:
            return False

    def _is_good_train(self, window=8):
        selected_loss = self._loss[-window:]
        loss_num = len(selected_loss)
        rise = 0
        for i in range(1, loss_num):
            if selected_loss[i] - selected_loss[i - 1] < 0:
                rise += 1

        if loss_num < 2:
            return True

        rise_rate = round(rise / (loss_num - 1), 4)

        return rise_rate >= 0.7

    def _ensemble_preds(self, reverse=False, k=5):
        """
        Get best val loss preds, best val acc preds,
        and mean to prevent overfitting.
        """
        all_preds_ids = list(self._all_preds.keys())
        top_k_preds_ids = sorted(all_preds_ids, key=lambda x: self._train_history[x], reverse=reverse)[:k]
        top_k_preds = [self._all_preds[i] for i in top_k_preds_ids]

        return np.mean(top_k_preds, axis=0)

    @timeit
    def train(self, remain_time_budget=None):
        log("model {} begin trainning".format(self._model.model_name))
        super(EnhancementStage, self).train(remain_time_budget)
        self._preprocess_data()
        epoch_num = self._decide_epoch_num()

        if self._stage_loop_num == 1:
            self._train_params["is_multilabel"] = self.ctx.is_multilabel
            early_stopping = TerminateOnBaseline(monitor="acc", baseline=0.999)
            self._train_params["callbacks"] = [early_stopping]
        else:
            self._train_params["callbacks"] = []

        history = self._model.fit(self._pre_train_x, self._pre_train_y,
                                  # validation_data_fit=(self._pre_val_x, self._pre_val_y),
                                  validation_data_fit=(None, None),
                                  epochs=epoch_num,
                                  cur_loop_num=self._stage_loop_num,
                                  params=self._train_params)

        train_loss = round(history.history.get('loss')[-1], 6)
        train_acc = round(history.history.get('acc')[-1], 6)
        lr = history.history.get('lr')
        log(f"stage_loop_num {self._stage_loop_num} train loss {train_loss} train acc {train_acc}")
        self._loss.append(train_loss)
        self._acc.append(train_acc)
        self._train_history[self._stage_loop_num] = train_loss

        if self.ctx.use_knn_fix:
            _pre_train_x = np.asarray(list(self._pre_train_x_dict.values()), dtype=np.float32)
            _pre_train_x = _pre_train_x[:, :, :, np.newaxis]
            _, y = self.ctx.raw_data
            _pre_train_y = np.asarray([y[i] for i in self._pre_train_x_dict.keys()], dtype=np.int32)
            log(f"x shape knn fix: {np.array(self._pre_train_x).shape}")
            self.ctx.train_y_true = _pre_train_y
            self.ctx.train_y_hat = self._model.predict(_pre_train_x)

    @timeit
    def test(self, remain_time_budget=None):
        super(EnhancementStage, self).train(remain_time_budget)

        if self._stage_test_loop_num == 0 or self._spec_len_status == 2:
            self._spec_len_status = 0
            self._feature_params['mode'] = 'test'
            x = get_features(self.ctx.raw_test_data, self._feature_params)
            x = np.array(x)
            self._pre_test_x = x[:, :, :, np.newaxis]
            log(
                f"stage_loop_num={self._stage_loop_num}, preprocess {len(self._pre_test_x)} test data, shape {self._pre_test_x.shape}")

        while self._stage_loop_num <= self._decide_warmup_loops():
            self.train(remain_time_budget=remain_time_budget)

        score = 0
        if self._decide_use_all_data() is False:
            score = balanced_acc_metric(self._pre_val_y, self._model.predict(self._pre_val_x))
            if (score - 0.01 > self.ctx.max_score < 0.90) or (score >= self.ctx.max_score >= 0.90):
                self._better_score_cnt += 1
            log("resnet score {} max_score {}".format(score, self.ctx.max_score))

        if self._is_predict() and self._decide_use_all_data():
            if self._is_ensenmble():
                if self._is_good_train():
                    preds = self._model.predict(self._pre_test_x, batch_size=8)
                    # normalize logits
                    # preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
                    self._all_preds[self._stage_loop_num] = preds
                    log("this round is good train")
                else:
                    log("this round is bad train")
                preds = self._ensemble_preds()
            else:
                preds = self._model.predict(self._pre_test_x, batch_size=8)
            self._last_pred = preds
            self._last_pred_loop = self._stage_loop_num
        elif self._decide_use_all_data() is False:
            if len(self.ctx.lr_last_preds) > 0:
                preds = self.ctx.lr_last_preds
            else:
                preds = self.ctx.ensemble_predicts()
        else:
            preds = self._last_pred

        self._stage_test_loop_num += 1
        if self._stage_loop_num >= self._stage_end_loop_num \
                or (
                self._stage_test_loop_num >= 8 and self._decide_use_all_data() is False and self.ctx.max_score > 0.4):
            self._need_transition = True
        if self._need_transition:
            self._transition()

        return preds

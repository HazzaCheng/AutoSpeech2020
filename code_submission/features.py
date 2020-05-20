#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-03-19
import os
from functools import partial
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

from CONSTANT import SR
from tools import timeit, pad_seq

NCPU = os.cpu_count() - 1
my_pool = ThreadPool(NCPU)

# ---------------- constants -----------------------
MFCC = 'mfcc'
ZERO_CROSSING_RATE = 'zero crossing rate'
SPECTRAL_CENTROID = 'spectral centroid'
MEL_SPECTROGRAM = 'mel spectrogram'
SPECTRAL_ROLLOFF = 'spectral rolloff'
CHROMA_STFT = 'chroma stft'
BANDWIDTH = 'bandwidth'
SPECTRAL_CONTRAST = 'spectral_contrast'
SPECTRAL_FLATNESS = 'spectral flatness'
TONNETZ = 'tonnetz'
CHROMA_CENS = 'chroma cens'
RMS = 'rms'
POLY_FEATURES = 'poly features'
STFT = 'stft'

NUM_MFCC = 96  # num of mfcc features, default value is 24
MAX_AUDIO_DURATION = 5  # limited length of audio, like 20s
AUDIO_SAMPLE_RATE = 16000
MAX_FRAME_NUM = 700
FFT_DURATION = 0.1
HOP_DURATION = 0.04

SPEECH_FEATURES = [MFCC,
                   ZERO_CROSSING_RATE,
                   SPECTRAL_CENTROID,
                   MEL_SPECTROGRAM,
                   SPECTRAL_ROLLOFF,
                   CHROMA_STFT,
                   BANDWIDTH,
                   SPECTRAL_CONTRAST,
                   SPECTRAL_FLATNESS,
                   TONNETZ,
                   CHROMA_CENS,
                   RMS,
                   POLY_FEATURES,
                   STFT]

# ---------------- parallel extract features -----------------------


def extract_parallel(data, extract):
    data_with_index = list(zip(data, range(len(data))))
    results_with_index = list(my_pool.map(extract, data_with_index))

    results_with_index.sort(key=lambda x: x[1])

    results = []
    for res, idx in results_with_index:
        results.append(res)

    return np.asarray(results)


# mfcc
@timeit
def extract_mfcc(data, sr=16000, n_mfcc=NUM_MFCC):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d, sr=sr, n_mfcc=n_mfcc)
        r = r.transpose()
        results.append(r)

    return results


def extract_for_one_sample(tuple, extract, use_power_db=False, **kwargs):
    data, idx = tuple
    data = data.astype(np.float32)
    r = extract(data, **kwargs)
    # for melspectrogram
    if use_power_db:
        r = librosa.power_to_db(r)

    r = r.transpose()
    return r, idx


@timeit
def extract_mfcc_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mfcc=NUM_MFCC):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.mfcc, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    results = extract_parallel(data, extract)

    return results


# zero crossings

@timeit
def extract_zero_crossing_rate_parallel(data):
    extract = partial(extract_for_one_sample, extract=librosa.feature.zero_crossing_rate, pad=False)
    results = extract_parallel(data, extract)

    return results


# spectral centroid

@timeit
def extract_spectral_centroid_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_centroid, sr=sr,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_melspectrogram_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mels=40, use_power_db=False):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.melspectrogram,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, use_power_db=use_power_db)
    results = extract_parallel(data, extract)

    return results


# spectral rolloff
@timeit
def extract_spectral_rolloff_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_rolloff,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)  # data+0.01?
    # sklearn.preprocessing.scale()
    return results


# chroma stft
@timeit
def extract_chroma_stft_parallel(data, sr=16000, n_fft=None, hop_length=None, n_chroma=12):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_stft, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_bandwidth_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_bandwidth,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_spectral_contrast_parallel(data, sr=16000, n_fft=None, hop_length=None, n_bands=6):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_contrast,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_spectral_flatness_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_flatness,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_tonnetz_parallel(data, sr=16000):
    extract = partial(extract_for_one_sample, extract=librosa.feature.tonnetz, sr=sr)
    results = extract_parallel(data, extract)
    return results


@timeit
def extract_chroma_cens_parallel(data, sr=16000, hop_length=None, n_chroma=12):
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_cens, sr=sr,
                      hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_rms_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.rms,
                      frame_length=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_poly_features_parallel(data, sr=16000, n_fft=None, hop_length=None, order=1):
    if n_fft is None:
        n_fft = int(sr * FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr * HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.poly_features,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, order=order)
    results = extract_parallel(data, extract)

    return results


# stft
def get_specified_feature_func(feature_name,
                               sr=16000,
                               n_fft=None,
                               hop_length=None,
                               n_mfcc=96,
                               n_mels=40,
                               use_power_db=True,
                               n_chroma=12,
                               n_bands=6,
                               order=1):
    if feature_name == MFCC:
        func = partial(extract_mfcc_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    elif feature_name == ZERO_CROSSING_RATE:
        func = partial(extract_zero_crossing_rate_parallel)
    elif feature_name == SPECTRAL_CENTROID:
        func = partial(extract_spectral_centroid_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == MEL_SPECTROGRAM:
        func = partial(extract_melspectrogram_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, use_power_db=use_power_db)
    elif feature_name == SPECTRAL_ROLLOFF:
        func = partial(extract_spectral_rolloff_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == CHROMA_STFT:
        func = partial(extract_chroma_stft_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    elif feature_name == BANDWIDTH:
        func = partial(extract_bandwidth_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == SPECTRAL_CONTRAST:
        func = partial(extract_spectral_contrast_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    elif feature_name == SPECTRAL_FLATNESS:
        func = partial(extract_spectral_flatness_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == TONNETZ:
        func = partial(extract_tonnetz_parallel, sr=sr)
    elif feature_name == CHROMA_CENS:
        func = partial(extract_chroma_cens_parallel, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    elif feature_name == RMS:
        func = partial(extract_rms_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_name == POLY_FEATURES:
        func = partial(extract_poly_features_parallel, sr=sr, n_fft=n_fft, hop_length=hop_length, order=order)
    else:
        raise Exception("No such feature {}".format(feature_name))

    return func


# TODO stft need foramt
def extend_wav(wav, train_wav_len=40000, test_wav_len=40000, mode='train'):
    if mode == 'train' or mode == 'fast_train':
        div, mod = divmod(train_wav_len, wav.shape[0])
        extended_wav = np.concatenate([wav] * div + [wav[:mod]])
        # reverse
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        div, mod = divmod(test_wav_len, wav.shape[0])
        extended_wav = np.concatenate([wav] * div + [wav[:mod]])
        return extended_wav

def get_stft_from_wav(wav, hop_length, win_length, n_fft):
    data = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    return data.T


def get_mfcc_from_wav(wav, sr, hop_length, win_length, n_fft, n_mfcc, n_mels):
    data = librosa.feature.mfcc(wav, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mfcc=n_mfcc, n_mels=n_mels)
    return data.T


def get_mel_from_wav(wav, sr, hop_length, win_length, n_fft, n_mels):
    data = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)
    r = librosa.power_to_db(data)
    return r.T


def load_data(mag, feat_name=STFT, train_spec_len=250, test_spec_len=250, mode='train'):
    freq, time = mag.shape
    if mode == 'train':
        if time - train_spec_len > 0:
            randtime = np.random.randint(0, time - train_spec_len)
            spec_mag = mag[:, randtime:randtime + train_spec_len]
        else:
            spec_mag = mag[:, :train_spec_len]
    elif mode == 'fast_train':
        spec_mag = mag[:, :train_spec_len]
    else:
        # TODO test数据不需要截取
        spec_mag = mag[:, :test_spec_len]

    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    normal_mag = (spec_mag - mu) / (std + 1e-5)
    if feat_name == STFT:
        return normal_mag
    else:
        return normal_mag.T


def pad_data_by_copy(data, spec_len, mode="train"):
    time, freq = data.shape
    div, mod = divmod(spec_len, time)
    if mode == "train":
        if time > spec_len:
            randtime = np.random.randint(0, time - spec_len)
            spec_data = data[randtime: randtime + spec_len, :]
        else:
            spec_data = np.concatenate([np.tile(data.transpose(), div).transpose(), data[:mod, :]], axis=0)
    else:
        if time > spec_len:
            spec_data = data[:spec_len, :]
        else:
            spec_data = np.concatenate([np.tile(data.transpose(), div).transpose(), data[:mod, :]], axis=0)
    return spec_data


def wav_to_mag(wav, params):
    mode = params["mode"]
    train_wav_len = params["train_wav_len"]
    test_wav_len = params["test_wav_len"]
    feature = params["feature"]

    wav = extend_wav(wav, train_wav_len, test_wav_len, mode=mode)
    wav = np.asfortranarray(wav, dtype=np.float32)

    if feature == STFT:
        data = get_stft_from_wav(wav, hop_length=params["hop_length"], win_length=params["win_length"],
                                 n_fft=params["n_fft"])
        mag, _ = librosa.magphase(data)
        mag_T = mag.T
    elif feature == MFCC:
        data = get_mfcc_from_wav(wav, sr=params["sr"], n_mfcc=params['n_mfcc'], hop_length=params["hop_length"],
                                 win_length=params["win_length"], n_fft=params["n_fft"], n_mels=params['n_mels'])
        mag_T = data.T
    elif feature == MEL_SPECTROGRAM:
        data = get_mel_from_wav(wav, sr=params["sr"], n_mels=params['n_mels'], hop_length=params["hop_length"],
                                win_length=params["win_length"], n_fft=params["n_fft"])
        mag_T = data.T
    else:
        raise Exception("No such feature")
    if mode == 'test':
        mag_T = load_data(mag_T, feature, params["train_spec_len"], params["test_spec_len"], mode)

    return mag_T

@timeit
def get_features(wav_list, params):
    if len(wav_list) == 0:
        return []
    elif len(wav_list) > NCPU * 8:
        with Pool(NCPU) as pool:
            mag_arr = pool.starmap(wav_to_mag, zip(wav_list, repeat(params)))
            pool.close()
            pool.join()
            return mag_arr
    else:
        mag_arr = [wav_to_mag(wav, params) for wav in wav_list]
        return mag_arr

import keras
import numpy as np

from features import MFCC, MEL_SPECTROGRAM


class ModelSequenceDataGenerator(keras.utils.Sequence):
    def __init__(self, x, y,
                 mp_pooler, spec_len=250, batch_size=32, feat_name=MFCC,
                 shuffle=True, normalize=True, **kwargs):
        self.feat_name = feat_name
        self.spec_len = spec_len
        self.normalize = normalize
        self.mp_pooler = mp_pooler
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = np.arange(len(x))

        self.x = x
        self.y = y
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_x = [self.x[i] for i in indexes]
        x, y = self.__data_generation_mp(temp_x, indexes)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_mp(self, x_temp, indexes):

        x = [self.mp_pooler.apply_async(data_augment, args=(mag, "train")) for mag in x_temp]
        x = np.array([p.get() for p in x])
        y = self.y[indexes]
        return x, y


def data_augment(mag, mode):
    if np.random.rand() < 0.3:
        mag = mag[::-1, :]
    # mu = np.mean(mag, 1, keepdims=True)
    # std = np.std(mag, 1, keepdims=True)
    # normal_mag = (mag - mu) / (std + 1e-5)
    normal_mag = mag
    return normal_mag

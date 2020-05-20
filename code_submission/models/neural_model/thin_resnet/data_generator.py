import keras
import numpy as np

from features import load_data, STFT, MFCC


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, dim,
                 mp_pooler, spec_len=250, batch_size=32, feat_name=STFT,
                 num_classes=None, augmentation=True, shuffle=True, normalize=True, 
                 nfft=512, win_length=400, hop_length=160, **kwargs):
        # self.dim = dim
        # self.nfft = nfft
        self.feat_name = feat_name
        self.spec_len = spec_len
        # self.normalize = normalize
        self.mp_pooler = mp_pooler
        # self.win_length = win_length
        # self.hop_length = hop_length

        self.y = y
        self.shuffle = shuffle
        self.x = x
        # self.n_classes = num_classes
        self.batch_size = batch_size
        # self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_x = [self.x[i] for i in indexes]
        x, y = self.__data_generation_mp(temp_x, indexes)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_mp(self, x_temp, indexes):
        # load data 随机截取一段音频
        x = [self.mp_pooler.apply_async(load_data, args=(mag, self.feat_name, self.spec_len)) for mag in x_temp]
        x = np.array([p.get() for p in x])
        x = np.expand_dims(x, -1)
        y = self.y[indexes]
        return x, y

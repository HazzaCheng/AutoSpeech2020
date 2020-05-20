import numpy as np
import keras
from kapre.time_frequency import Melspectrogram
from sklearn.preprocessing import StandardScaler
from tools import log, timeit

@timeit
def get_fixed_array(X_list, len_sample):
    for i in range(len(X_list)):
        if len(X_list[i]) < len_sample:
            n_repeat = np.ceil(len_sample / X_list[i].shape[0]).astype(np.int32)
            X_list[i] = np.tile(X_list[i], n_repeat)

        X_list[i] = X_list[i][: len_sample]

    X = np.asarray(X_list)
    X = np.stack(X)
    X = X[:, :, np.newaxis]
    X = X.transpose(0, 2, 1)
    return X


def mel_feats_transform(x_mel):
    x_feas = []
    for i in range(len(x_mel)):
        mel = np.mean(x_mel[i], axis=0).reshape(-1)
        mel_std = np.std(x_mel[i], axis=0).reshape(-1)
        fea_item = np.concatenate([mel, mel_std], axis=-1)
        x_feas.append(fea_item)

    x_feas = np.asarray(x_feas)
    scaler = StandardScaler()
    X = scaler.fit_transform(x_feas[:, :])
    return X


class KapreMelSpectroGramFeatsMaker:
    SAMPLING_RATE = 16000
    N_MELS = 30
    HOP_LENGTH = int(SAMPLING_RATE * 0.04)
    N_FFT = 1024
    FMIN = 20
    FMAX = SAMPLING_RATE // 2

    # CROP_SEC = 5

    def __init__(self, cut_length):
        self.kapre_melspectrogram_extractor = None
        self.kape_params = {
            "SAMPLING_RATE": self.SAMPLING_RATE,
            "N_MELS": self.N_MELS,
            "HOP_LENGTH": int(self.SAMPLING_RATE * 0.04),
            "N_FFT": self.N_FFT,
            "FMIN": self.FMIN,
            "FMAX": self.SAMPLING_RATE // 2,
            "CUT_LENGTH": cut_length
        }
        self.init_kapre_melspectrogram_extractor()

    def make_melspectrogram_extractor(self, input_shape, sr=SAMPLING_RATE):
        model = keras.models.Sequential()
        model.add(
            Melspectrogram(
                fmax=self.kape_params.get("FMAX"),
                fmin=self.kape_params.get("FMIN"),
                n_dft=self.kape_params.get("N_FFT"),
                n_hop=self.kape_params.get("HOP_LENGTH"),
                n_mels=self.kape_params.get("N_MELS"),
                name="melgram",
                image_data_format="channels_last",
                input_shape=input_shape,
                return_decibel_melgram=True,
                power_melgram=2.0,
                sr=sr,
                trainable_kernel=False,
            )
        )
        return model

    def init_kapre_melspectrogram_extractor(self):
        # self.kapre_melspectrogram_extractor = self.make_melspectrogram_extractor(
        #     (1, self.kape_params.get("CROP_SEC") * self.kape_params.get("SAMPLING_RATE"))
        # )
        # if KAPRE_FMAKER_WARMUP:
        #     warmup_size = 10
        #     warmup_x = [
        #         np.array([np.random.uniform() for i in range(48000)], dtype=np.float32) for j in range(warmup_size)
        #     ]
        #     warmup_x_mel = self.make_features(warmup_x, feats_maker_params={"len_sample": 5, "sr": 16000})

        self.kapre_melspectrogram_extractor = self.make_melspectrogram_extractor(
            (1, self.kape_params.get("CUT_LENGTH"))
        )

    @timeit
    def make_features(self, raw_data):
        # raw_data = [sample[0 : MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE] for sample in raw_data]
        # X = get_fixed_array(raw_data, len_sample=feats_maker_params.get("len_sample"), sr=feats_maker_params.get("sr"))

        X = get_fixed_array(raw_data, self.kape_params.get("CUT_LENGTH"))

        X = self.kapre_melspectrogram_extractor.predict(X)

        X = np.squeeze(X)
        X = X.transpose(0, 2, 1)

        X = mel_feats_transform(X)
        return X
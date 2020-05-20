from models.neural_model.attention_gru import AttentionGruModel
from models.neural_model.bilstm_attention import BilstmAttentionModel
from models.neural_model.cnn import Cnn2DModel
from models.neural_model.crnn2d import Crnn2dModel
from models.neural_model.lstm_attention import LstmAttentionModel
from models.neural_model.thin_resnet.thin_resnet import ThinResnet
from models.simple_model.logistic_regression import LogisticRegression

# MODEL NAME
CNN2D_MODEL = 'cnn2d'
CRNN2D_MODEL = 'crnn2d'
SVM_MODEL = 'svm'
BILSTM_MODEL = 'bilstm'
LSTM_MODEL = 'lstm'
LR_MODEL = 'lr'
ATT_GRU_MODEL = 'att_gru'
THINRESNET_MODEL = 'thin_resnet'


SPEECH_MODEL_LIB = {
    LR_MODEL: LogisticRegression,
    LSTM_MODEL: LstmAttentionModel,
    CRNN2D_MODEL: Crnn2dModel,
    BILSTM_MODEL: BilstmAttentionModel,
    CNN2D_MODEL: Cnn2DModel,
    ATT_GRU_MODEL: AttentionGruModel,
    THINRESNET_MODEL: ThinResnet
}

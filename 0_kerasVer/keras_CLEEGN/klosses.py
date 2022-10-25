from tensorflow import keras
from tensorflow.keras import backend as K

def RootMeanSquaredError(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def MeanLogSquaredError(y_true, y_pred):
    return K.mean(K.log(1.0 + K.square(y_pred - y_true)))

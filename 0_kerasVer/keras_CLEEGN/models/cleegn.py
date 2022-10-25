""" CLEEGN model
2022/01/22
"""
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import numpy as np
import sys
import os

def CLEEGN(n_chan=64, duration=128, N_F=20, fs=128.0):
    inputs = layers.Input(shape=(n_chan, duration, 1))
    block1 = layers.Conv2D(n_chan, (n_chan, 1), padding="valid", use_bias=True)(inputs)
    block1 = layers.Permute((3, 2, 1))(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Conv2D(N_F, (1, int(fs * 0.1)), use_bias=True, padding="same")(block1)
    block1 = layers.BatchNormalization()(block1)

    block2 = layers.Conv2D(N_F, (1, int(fs * 0.1)), use_bias=True, padding="same")(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Conv2D(n_chan, (n_chan, 1), padding="same", use_bias=True)(block2)
    block2 = layers.BatchNormalization()(block2)

    outputs = layers.Conv2D(1, (n_chan, 1), padding="same")(block2)
    return models.Model(inputs=inputs, outputs=outputs)

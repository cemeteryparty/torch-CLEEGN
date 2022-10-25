""" CLEEGN model
2022/03/24
"""
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import backend as K
import numpy as np
import math
import sys
import os

""" SCCNet """
def square(x):
    """
    square activation function
    """
    return K.square(x)
def SCCNet(nb_classes, Chans=64, Samples=128, fs=128, dropoutRate=0.5):
    inputs = layers.Input(shape=(Chans, Samples, 1))
    block1 = layers.Conv2D(Chans, (Chans, 1), padding="valid", use_bias=True)(inputs)
    block1 = layers.Permute((3, 2, 1))(block1)
    block1 = layers.BatchNormalization()(block1)
    
    block2 = layers.Conv2D(20, (1, 12), use_bias=True, padding="same")(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation(square)(block2) # block2 = Lambda(lambda x: x ** 2)(block2)
    block2 = layers.Dropout(rate=dropoutRate)(block2)

    block3 = layers.AveragePooling2D(
        pool_size=(1, math.ceil(0.5 * fs)),
        strides=(1, math.ceil(0.1 * fs))
    )(block2) # default: padding="valid"
    
    block4 = layers.Flatten()(block3)
    outputs = layers.Dense(units=nb_classes, use_bias=True, activation="softmax")(block4)

    return models.Model(inputs=inputs, outputs=outputs)

def EEGNet(nb_classes, Chans=64, Samples=128,
    dropoutRate=0.5, kernLength=64, F1=8, D=2,
    F2=16, norm_rate=0.25, dropoutType="Dropout"):
    
    if dropoutType == "SpatialDropout2D":
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
            'or Dropout, passed as a string.')
    
    input1 = layers.Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = layers.Conv2D(
        F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1),
        use_bias=False
    )(input1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.DepthwiseConv2D(
        (Chans, 1), use_bias=False, depth_multiplier=D, 
        depthwise_constraint=constraints.max_norm(1.)
    )(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('elu')(block1)
    block1 = layers.AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    
    block2 = layers.SeparableConv2D(
        F2, (1, 16), use_bias=False, padding='same'
    )(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)
        
    flatten = layers.Flatten(name='flatten')(block2)
    
    dense = layers.Dense(
        nb_classes, name='dense', kernel_constraint=constraints.max_norm(norm_rate)
    )(flatten)
    softmax = layers.Activation('softmax', name='softmax')(dense)
    
    return models.Model(inputs=input1, outputs=softmax)

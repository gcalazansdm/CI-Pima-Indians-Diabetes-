from keras.models import Model

from keras.layers import Input
from keras.layers import Dense

from keras.optimizers import Adam
from keras.layers import Activation

import matplotlib.pyplot as plt
import numpy as np

import math
import datetime
import os

def step(x):
	if x < 0:
		return 0.0
	else:
		return 0.5

def create_network():
    input_layer = Input(shape=(8,))

    # camada escondida
    hidden_layer = Dense(units=5,
                         activation='relu')(input_layer)

    # camada de saida
    output_layer = Dense(units=2,activation='softmax')(hidden_layer)

    classifier = Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=Adam(0.005), loss='categorical_crossentropy', metrics=['accuracy','categorical_accuracy'])

    return classifier
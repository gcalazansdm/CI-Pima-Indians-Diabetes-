from keras.models import Model

from keras.layers import Input
from keras.layers import Dense

from keras.optimizers import Adam

def create_network():
    input_layer = Input(shape=(8,))

    # camada escondida
    hidden_layer = Dense(units=5,
                         activation='relu')(input_layer)
    # camada encode
    hidden_layer = Dense(units=3,
                         activation='relu')(input_layer)

    # camada escondida
    hidden_layer = Dense(units=5,
                         activation='relu')(input_layer)
    # camada de saida
    output_layer = Dense(units=8,activation='relu')(hidden_layer)

    classifier = Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=Adam(0.005), loss='MSE', metrics=['MAE','MSE'])

    return classifier
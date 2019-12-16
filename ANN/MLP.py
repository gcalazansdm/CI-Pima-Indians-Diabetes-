from ANN.ANN import ANN

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense

from keras.optimizers import Adam

class MLP(ANN):
    def __init__(self,extra = ""):
        super().__init__()
        self.name = "MLP" + extra

    def create_network(self):
        input_layer = Input(shape=(8,))

        # camada escondida
        hidden_layer = Dense(units=5,
                             activation='relu')(input_layer)

        # camada de saida

        output_layer = Dense(units=2,activation='softmax')(hidden_layer)

        classifier = Model(inputs=input_layer, outputs=output_layer)
        classifier.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy','categorical_accuracy'])

        return classifier

    def test(self, test_base):
        return self.test_ann(test_base)

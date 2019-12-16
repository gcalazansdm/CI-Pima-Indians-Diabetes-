from ANN.ANN import ANN
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.regularizers import l2

class SVM(ANN):
    def __init__(self,extra = ""):
        super().__init__()
        self.name = "SVM" + extra

    def create_network(self):
        input_layer = Input(shape=(8,))

        # camada escondida
        hidden_layer = Dense(64, activation='relu')(input_layer)

        # camada de saida
        output_layer = Dense(units=2,activation='softmax', kernel_regularizer= l2(0.01))(hidden_layer)

        classifier = Model(inputs=input_layer, outputs=output_layer)
        classifier.compile(loss='hinge',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        return classifier

    def test(self, test_base):
        return self.test_ann(test_base)
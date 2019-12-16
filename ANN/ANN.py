from keras.callbacks.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import os
class ANN():
    def __init__(self):
        self.name = "ANN"
        self.network = self.create_network()
    def create_network(self):
        pass

    def useNetwork(self,train_base,val_base,epochs=10000,patience=1000):
        if os.path.isfile(self.name + '.hf5'):
            self.network.load_weights(self.name + ".hf5")
        else:
            Model = ModelCheckpoint(self.name + ".hf5")
            Early = EarlyStopping(patience=1000)
            self.network.fit(x=train_base[0], y=train_base[1], batch_size=20, epochs=epochs, callbacks=[Early, Model],
                             validation_split=0.1, validation_data=val_base,
                            shuffle=True, use_multiprocessing=True)

    def predict(self,test_base):
        return self.network.predict(test_base)

    def test_ann(self,test_base,acuraccy=False,matrix=False):
        results = self.predict(test_base[0])
        x,y = results.shape
        print(test_base[1].shape)
        rValue = None
        if(acuraccy):
            rValue = np.mean(np.power(np.subtract(test_base[1],results), 1))
        else:
            if(matrix):
                rValue = [[0, 0], [0, 0],[0,0]]

                for i in range(0,x):
                    if(results[i][0] > results[i][1] ):
                        rValue[0][0] += test_base[1][i][0]
                        rValue[2][0] += test_base[1][i][0]
                        rValue[2][1] += 1
                        rValue[0][1] += test_base[1][i][1]
                    else:
                        rValue[1][0] += test_base[1][i][0]
                        rValue[1][1] += test_base[1][i][1]
                        rValue[2][0] += test_base[1][i][1]
                        rValue[2][1] += 1
            else:
                rValue = 0
                for i in range(0,x):
                    if(results[i][0] > results[i][1] ):
                        rValue += test_base[1][i][0]
                    else:
                        rValue += test_base[1][i][1]
                rValue /= x
        return rValue

from keras.callbacks.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import keras.utils
import MLP
import KNN


def readcsv(filename):
    original_csv = pd.read_csv(filename)
    return original_csv
def filterdata(dataset):
    rValue = dataset.copy(deep = True)
    rValue[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Age']] = dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Age']].replace(0, np.NaN)
    return rValue.dropna()
dataset = readcsv('diabetes.csv')
#print(dataset)
#print(filterdata(dataset))
dataset_normal = (dataset - dataset.min()) / (dataset.max() - dataset.min())
print(dataset_normal)
labels = keras.utils.to_categorical(dataset_normal[['Outcome']])
values = dataset_normal.drop(columns="Outcome")
X_train,X_test,y_train,y_test = train_test_split(values,labels,test_size=0.3)
D_train,V_test,D_y_train,V_y_test = train_test_split(X_test,y_test,test_size=0.7)

netWork = MLP.create_network()
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
Model = ModelCheckpoint("elem")
Early = EarlyStopping(patience=1000)


#netWork.fit(x=X_train, y=y_train, batch_size=20, epochs=1000000, callbacks=[Early,Model], validation_split=0.1, validation_data=(X_test,y_test), shuffle=True, use_multiprocessing=True)
KNN.trainKNN(X_train,y_train,X_test,y_test)
#MLP.train(netWork, (X_train,y_train), (D_train,D_y_train), (V_test,V_y_test), num_epochs_without_change=100, printing=True)

#.to_numpy()

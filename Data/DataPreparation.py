import numpy as np
from sklearn.model_selection import train_test_split
import keras.utils

def filterdata(dataset):
    rValue = dataset.copy(deep = True)
    rValue[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Age']] = dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Age']].replace(0, np.NaN)
    return rValue.dropna()

def dataSeparation(dataset):
    labels = keras.utils.to_categorical(dataset[['Outcome']])
    values = dataset.drop(columns="Outcome")
    X_train, X_test, y_train, y_test = train_test_split(values, labels, test_size=0.3)
    D_train, V_test, D_y_train, V_y_test = train_test_split(X_test, y_test, test_size=0.7)
    return (X_train,y_train),(D_train,D_y_train),(V_test,V_y_test)

def to_numpy(base):
    return base[0].to_numpy(), base[1]
def normalize(dataset):
    dataset_normal = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    return dataset_normal
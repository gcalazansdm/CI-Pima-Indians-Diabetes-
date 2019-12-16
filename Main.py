
import Data
import ANN
import random

dataset = Data.readcsv('diabetes.csv')
fdataset = Data.filterdata(dataset)

normalized_filtred_dataset = Data.normalize(fdataset)
train,dev,val = Data.dataSeparation(normalized_filtred_dataset)

train = Data.to_numpy(train)
dev = Data.to_numpy(dev)
val = Data.to_numpy(val)
encoder_values = train[0].copy()
print(encoder_values)
encoder_labels = train[0].copy()
dev_encoder_values = dev[0].copy()
dev_encoder_labels = dev[0].copy()
x,y = encoder_values.shape
for i in range(0,x):
    encoder_values[i][random.randint(0,y-1)] = 0
x,y = dev_encoder_values.shape
for i in range(0,x):
    dev_encoder_values[i][random.randint(0,y-1)] = 0

Autoencoder = ANN.AutoEncoder()
Autoencoder.useNetwork((encoder_values,encoder_labels),(dev_encoder_values,dev_encoder_labels))

SVM = ANN.SVM()
SVM.useNetwork(train,dev,patience=100)

mlp = ANN.MLP()
mlp.useNetwork(train,dev)

GAN = ANN.GAN()
GAN.useNetwork(train,dev)

normalized_dataset = Data.normalize(dataset)
train,dev,val = Data.dataSeparation(normalized_dataset)

train = Data.to_numpy(train)
dev = Data.to_numpy(dev)
val = Data.to_numpy(val)

EncodedTrain = Autoencoder.predict(train[0])
train = (EncodedTrain,train[1])
EncodedDev = Autoencoder.predict(dev[0])
dev = (EncodedDev,dev[1])

SVM_saved = ANN.SVM("_auto")
SVM_saved.useNetwork(train,dev,patience=100)

mlp_saved = ANN.MLP("_auto")
mlp_saved.useNetwork(train,dev)

print(Autoencoder.test((val[0],val[0])))

print(SVM.test(val))
print(mlp.test(val))

print(SVM_saved.test(val))
print(mlp_saved.test(val))
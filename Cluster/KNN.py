from sklearn.neighbors import KNeighborsClassifier
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

class KNN:
    def __init__(self,n_neighbors=5):
       self.classifier = KNeighborsClassifier(n_neighbors)

    def train(self,images,labels):
        self.classifier.fit(images, labels)

    def predict(self,X_test):
        pred_i = self.classifier.predict(X_test)
        return pred_i
    def save(self):
        with open('KNNFile.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

def loadKNN():
    with open('KNNFile.pkl', 'rb') as input:
        return pickle.load(input)

def trainKNN(values, labels, yvalues,ylabels):
    error = []
    classifier = KNN(5)
    classifier.train(values, labels)
    classifier.save()
    for i in range(1, 40):
        classifier = KNN(i)
        classifier.train(values, labels)
        predicted = classifier.predict(yvalues)
        error.append(np.mean(np.power(predicted != ylabels,2)))
        print(i,np.mean(predicted != ylabels))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('MSE')
    plt.show()
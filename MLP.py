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



def calculateErrorPerNeuron(values, indexes):
    mse_error = np.mean(np.power(values - indexes, 2))
    mape_error = np.mean(np.divide(np.absolute(values - indexes), np.absolute(indexes)) * 100)

    return [mse_error, mape_error]


def test_network(base, network):
    normalized_values = base[0]
    labels = base[1]
    results = network.predict(normalized_values)
    test_loss = calculateErrorPerNeuron(results, labels)
    return test_loss

def randomize(dataset, labels):
    # Generate the permutation index array.
    permutation = np.random.permutation(dataset.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def save_weights(model, name):
    file_ = str(name) + ".h5"
    directory = os.path.split(file_)[0]

    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save_weights(str(name) + ".h5")


def trysave(test_loss, network, epoch, best_loss, printing=False):
    targetloss = test_loss[0]
    rLoss = best_loss
    if (targetloss > 0. and best_loss > targetloss):
        save_weights(network, "./net")
        if (printing):
            print("Saving, Old Value: " + str(best_loss) + " & New value: " + str(targetloss) + " gain : " + str(
                abs(best_loss - targetloss)))
        rLoss = targetloss
    elif (printing):
        print("Not saving, Old Value: " + str(best_loss) + " & New value: " + str(targetloss) + " lost : " + str(
            abs(targetloss - best_loss)))
    return rLoss

def train(network, train_base, test_base, val_base, batch_size=20, num_epochs_without_change=100, printing=True):
    start_time = datetime.datetime.now()
    best_loss = np.inf
    Stop = False
    h = 0
    num_error_starvation = 0
    lenBase = len(train_base[0])
    maxLen = math.ceil(lenBase / float(batch_size))
    labels = ["MSE", "MAPE"]
    oldLabels = []
    oldValLabels = []
    oldTrainLabels = []
    for i in range(0, len(labels) +1):
        oldLabels.append([])
        oldValLabels.append([])
        oldTrainLabels.append([])
    directory = "."
    if not os.path.exists(directory):
        os.makedirs(directory)

    while (not Stop):

        epoch_start = datetime.datetime.now()

        train_elems = train_base[0]
        train_labels = train_base[1]

        randomize(train_elems, train_labels)

        loss = [0, 0]

        for x in range(0, maxLen):
            x_train = train_elems[batch_size * x:batch_size * (x + 1)]
            x_test = train_labels[batch_size * x:batch_size * (x + 1)]

            train_loss = network.train_on_batch(x_train, x_test)
            for index in range(0, len(train_loss) - 1):
                loss[index] += train_loss[index + 1]


        test_loss = network.test_on_batch(test_base[0],test_base[1])
        for i in range(0, len(test_loss)):
            oldLabels[i].append(test_loss[i])

        val_loss = network.test_on_batch(val_base[0],val_base[1])
        print(val_loss)
        for i in range(0, len(val_loss)):
            oldValLabels[i].append(val_loss[i])

        if (printing):
            print("Treino -> %d  Epoch" % (h + 1),oldValLabels[i])
        newError = trysave(val_loss, network, h, best_loss, printing)
        if (newError == best_loss):
            num_error_starvation += 1
        else:
            best_loss = newError
            num_error_starvation = 0

        elapsed_time = datetime.datetime.now() - start_time
        epoch_time = datetime.datetime.now() - epoch_start
        if (printing):
            print("\tnum no changes %d\n\ttotal time: %s %s" % (num_error_starvation, epoch_time, elapsed_time))

        for i in range(0, len(labels)):
            plt.plot(loss[i], 'r', label='Treino')
            plt.plot(oldLabels[i], 'b', label='Teste')
            plt.plot(oldValLabels[i], 'g', label='Teste')
            plt.ylabel(labels[i])
            plt.savefig(os.path.join(".", labels[i] + '.png'))
            plt.clf()
        h += 1
        Stop = num_error_starvation >= num_epochs_without_change
    return best_loss, h
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
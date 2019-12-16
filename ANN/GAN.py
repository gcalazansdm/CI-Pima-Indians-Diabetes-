from ANN.ANN import ANN

from tqdm import tqdm
import numpy as np
import os

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU

from keras.optimizers import Adam

class GAN(ANN):
    def __init__(self,extra = ""):
        super().__init__()
        self.name = "GAN" + extra


    def create_network(self):
        input_layer = Input(shape=(8,))
        d1 = Dense(units=24)(input_layer)
        act1  = LeakyReLU(0.2)(d1)
        drop1 = Dropout(0.3)(act1)

        d2 = Dense(units=16)(drop1)
        act2 = LeakyReLU(0.2)(d2)
        drop2 = Dropout(0.3)(act2)

        output_layer = Dense(units=8, activation='sigmoid')(drop2)
        network = Model(inputs=input_layer, outputs=output_layer)
        network.compile(loss='binary_crossentropy',metrics=['accuracy','categorical_accuracy'], optimizer=Adam(lr=0.0002, beta_1=0.5))
        return network
    def create_discriminator(self):
        input_layer = Input(shape=(8,))
        d1 = Dense(units=24)(input_layer)
        act1  = LeakyReLU(0.2)(d1)
        drop1 = Dropout(0.3)(act1)

        d2 = Dense(units=16)(drop1)
        act2  = LeakyReLU(0.2)(d2)
        drop2 = Dropout(0.3)(act2)

        d3 = Dense(units=16)(drop2)
        act3  = LeakyReLU(0.2)(d3)
        output_layer  = Dense(units=1, activation='sigmoid')(act3)
        discriminator = Model(inputs=input_layer, outputs=output_layer)

        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return discriminator

    def create_gan(self,discriminator, generator):
        discriminator.trainable = False
        gan_input = Input(shape=(8,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan

    def useNetwork(self,train_base,val_base,epochs=10000):
        if os.path.isfile(self.name + '.hf5'):
            self.network.load_weights(self.name + ".hf5")
        else:
            self.training(train_base,epochs)
    def training(self,train_data, epochs=1000000, batch_size=128):
        loss = -np.inf
        num_epochs = 0
        batch_count = train_data[0].shape[0] / batch_size
        X_train = train_data[0]
        discriminator = self.create_discriminator()
        gan = self.create_gan(discriminator, self.network)

        for e in range(1, epochs + 1):
            print("Epoch %d" % e)
            for _ in tqdm(range(batch_size)):
                # generate  random noise as an input  to  initialize the  generator
                noise = np.random.normal(0, 1, [batch_size, 8])

                generated_data = self.predict(noise)

                # Get a random set of  real images
                image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size)]

                # Construct different batches of  real and fake data
                X = np.concatenate([image_batch, generated_data])

                # Labels for generated and real data
                y_dis = np.zeros(2 * batch_size)
                y_dis[:batch_size] = 0.9

                # Pre train discriminator on  fake and real data  before starting the gan.
                discriminator.trainable = True
                discriminator.train_on_batch(X, y_dis)

                # Tricking the noised input of the Generator as real data
                noise = np.random.normal(0, 1, [batch_size, 8])
                y_gen = np.ones(batch_size)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                # We can enforce that by setting the trainable flag
                discriminator.trainable = False

                # training  the GAN by alternating the training of the Discriminator
                # and training the chained GAN model with Discriminator’s weights freezed.
                gan.train_on_batch(noise, y_gen)

                noise = np.random.normal(0, 1, [batch_size, 8])
                y_gen = np.ones(batch_size)

                this_loss = gan.test_on_batch(noise, y_gen)

                if(this_loss > loss):
                    loss = this_loss
                    self.network.save_weights(self.name+".hf5")
                    num_epochs  = -1
                num_epochs += 1
                if(num_epochs > 1000):
                    break
            if(num_epochs > 1000):
                break

    def predict(self,test_base):
        noise = np.random.normal(0, 1, [test_base[0].shape[0], 8])
        generated_data = self.predict(noise)
        return generated_data


import os
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #delete this if you want to use GPU

def load_data(data_path):
    '''
    Read images to numpy array
    '''
    files = os.listdir(data_path)

    data = []
    for file in files:
        img = Image.open(data_path + file)
        img.thumbnail((64, 64, ), PIL.Image.ANTIALIAS)
        img = img.convert('RGB')
        data.append(np.array(img))
    data = np.array(data)
    return data
    

class DCGAN:
    def __init__(self, data, epochs, batch_size):
        self.image_size = data.shape[1:]
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.adversarial = self.get_adversarial(self.discriminator, self.generator)
        self.data_generator = ImageDataGenerator()
        self.image_generator = self.data_generator.flow(data, batch_size=self.batch_size)
        
        
    def get_discriminator(self):
        '''
        Create discriminator model
        '''
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(5, 5),
                                 strides=(2, 2), padding='same',
                                 input_shape=self.image_size))
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(filters=128, kernel_size=(5, 5),
                                 strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(filters=256, kernel_size=(5, 5),
                                 strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(filters=512, kernel_size=(5, 5),
                                 strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        optimizer = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy',
                              optimizer=optimizer,
                              metrics=None)
        
        return model
        
    def get_generator(self):
        '''
        Create generator model
        '''
        model = Sequential()
        model.add(Dense(units=4 * 4 * 512,
                            input_shape=(1, 1, 100)))
        model.add(Reshape(target_shape=(4, 4, 512)))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters=256, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters=64, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters=3, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same'))
        model.add(Activation('tanh'))

        optimizer = Adam(lr=0.00015, beta_1=0.5)
        model.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=None)

        return model
      
     
    def get_adversarial(self, discriminator, generator):
        '''
        Create adversarial model
        '''
        model = Sequential()
        
        discriminator.trainable = False
        model.add(generator)
        model.add(discriminator)
        
        optimizer = Adam(lr=0.00015, beta_1=0.5)
        model.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=None)
        
        return model
    
    def save_model(self):
        '''
        Save all models into files
        '''
        self.discriminator.trainable = True
        self.discriminator.save('discriminator.h5')
        self.generator.save('generator.h5')
        self.adversarial.save('gan.h5')
    
    def train(self):
        batches_number = 150
        a_losses = []
        d_losses = []
        for epoch in range(self.epochs):
            for batch in range(batches_number):
                real_images = self.image_generator.next()
                real_images = (real_images - 127.5) / 127.5 #normalize data to [-1,1] range
                current_batch_size = real_images.shape[0]   #size of current batch
                real_labels = np.ones(current_batch_size) - np.random.random_sample(current_batch_size) * 0.2 #add some noise to labels
                
                noise = np.random.normal(0, 1, (current_batch_size, 1, 1, 100))
                fake_labels = np.zeros(current_batch_size) + np.random.random_sample(current_batch_size) * 0.2 #add some noise to labels
                fake_images = self.generator.predict(noise)
                
                #discriminator is not being trained during adversarial training
                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(real_images, real_labels)
                d_loss += self.discriminator.train_on_batch(fake_images, fake_labels)
                d_losses.append(d_loss)
                self.discriminator.trainable = False
                
                noise = np.random.normal(0, 1, (2*current_batch_size, 1, 1, 100))
                labels = np.ones(2*current_batch_size) - np.random.random_sample(2*current_batch_size) * 0.2 #add some noise to labels
                a_loss = self.adversarial.train_on_batch(noise, labels)
                a_losses.append(a_loss)
            print(f'Epoch:{epoch+1}\tDiscriminator losss: {d_losses[-1]}\tAdverserial loss:{a_losses[-1]}')
            
            noise = np.random.normal(0, 1, (1, 1, 1, 100))
            fake_image = self.generator.predict(noise)[0]
            plt.imshow(fake_image)
            plt.savefig('fake_image.jpg')
            
            if (epoch + 1) % 5 == 0:
                self.save_model()
                   

if __name__=='__main__':
    if len(sys.argv)==4:
        data_path = sys.argv[1] #path to images
        epochs = int(sys.argv[2]) #number of epochs
        batch_size = int(sys.argv[3]) #batch size
        data = load_data(data_path)
        gan = DCGAN(data, epochs, batch_size)
        gan.train()
    else:
        print("Wrong number of arguments!\nType 'python3 gan.py [path to images] [number of train epochs] [size of batches]'")

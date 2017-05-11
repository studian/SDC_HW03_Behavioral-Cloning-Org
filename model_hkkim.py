### Importing packages.

import os
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import scipy.misc
import random
from os import getcwd
from scipy.ndimage import rotate
from scipy.stats import bernoulli

from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Flatten
from keras.layers import Dropout, Lambda, ELU, Cropping2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras import initializations
from keras.models import model_from_json
from keras import backend as K
import json


def random_flip(image, steering_angle, flipping_prob=0.5):
    if random.random() < flipping_prob:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle

def random_brightness(image, median=0.8, dev=0.4):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = median + dev * np.random.uniform(-1.0, 1.0)
    hsv[:,:,2] = hsv[:,:,2]*random_bright

    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb

def new_img(image, steering_angle):
    image, steering_angle = random_flip(image, steering_angle)
    image = random_brightness(image)
    return image, steering_angle

STEERING_CORRECTION = 0.2

def get_next_images(batch_size=32):
    pos = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in pos:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_CORRECTION
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_CORRECTION
            image_files_and_angles.append((img, angle))

    return image_files_and_angles

def generator(batch_size=32):
    while True:
        x = []
        y = []
        images = get_next_images(batch_size)
        for img_file, steering_angle in images:
            image = plt.imread(DATA_PATH + img_file)
            angle = steering_angle
            new_image, new_angle = new_img(image, angle)
            x.append(new_image)
            y.append(new_angle)

        yield np.array(x), np.array(y)


# NVIDIA model
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

def get_model():

    input_shape = (160, 320, 3)
    #start = 0.001
    #stop = 0.001
    #nb_epoch = 10

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape,output_shape=input_shape))
    model.add(Cropping2D(cropping=((50,20),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Flatten())

    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dense(1))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=adam)
    return model

# Get the model
model = get_model()

# Train the model
#batch_size = 32
#number_of_samples_per_epoch = 25600

## Import data

DATA_PATH = './data/'

data = pd.read_csv(DATA_PATH + 'Udacity_driving_log.csv')
num_of_img = len(data)

# Examine data
print("Number of datapoints: %d" % num_of_img)

history_object = model.fit_generator(generator(batch_size=32),
                  samples_per_epoch = 25600,
                  nb_epoch = 2,
                  validation_data = generator(batch_size=32),
                  nb_val_samples = 5120,
                  verbose = 1)

### print the keys contained in the history object
print(history_object.history.keys())
print(model.summary())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('./results/Loss_of_training_and_validation.png')
plt.close()

print('Save the model')

model.save('model_hkkim.h5')
json_string = model.to_json()
with open('./model_hkkim.json', 'w') as f:
    f.write(json_string)

print('Done')

K.clear_session()

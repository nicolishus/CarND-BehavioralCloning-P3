# Model.py implements Nvidia's Self Driving Car CNN architecture with help
# from Udacity's walkthrough video for P3.

import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def get_csv(filepath):
    '''Function returns array with every line in the file as an element.'''
    lines_temp = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines_temp.append(line)
    return lines_temp

def create_model():
    '''Function that creates and returns the Keras model. It uses Nvidia's self driving car
    pipeline. It starts by normalizing the data to [0,1], centering to 0, and then cropping.
    This was done using keras so that the GPU could make these operations faster. The network
    then has 5 Convolutional layers, followed by a flatten layer, and 4 fully connected layers.
    I added a dropout layer after the largest fully connected layer so it wouldn't overfit the
    data.'''
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25),(0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def get_images_and_measurements(lines_array, correction=0.2):
    '''Function returns two arrays; one with the images and another with the measurements.
    It also takes in the left and right images and adds a correction value.'''
    images_temp = []
    measurements_temp = []
    for line in lines_array:
        for i in range(3):
            source_path = line[i]
            tokens = source_path.split('\\')    # my machine was windows based
            filename = tokens[-1]
            local_path = "./driving-data/IMG/" + filename
            image = cv2.imread(local_path)
            images_temp.append(image)
        measurement = float(line[3])
        measurements_temp.append(measurement)
        measurements_temp.append(measurement + correction)
        measurements_temp.append(measurement - correction)
    return images_temp, measurements_temp

def add_flipped_images(images_array, measurements_array):
    '''Function returns new arrays with original images/measurements and flipped
    images/opposite measurements to double the training data.'''
    flipped_images = []
    flipped_measurements = []
    for image, measurement in zip(images_array, measurements_array):
        flipped_images.append(image)
        flipped_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = measurement * -1.0
        flipped_images.append(flipped_image)
        flipped_measurements.append(flipped_measurement)
    return flipped_images, flipped_measurements

lines = get_csv("./driving-data/driving_log.csv")
images, measurements = get_images_and_measurements(lines, correction=0.2)
augmented_images, augmented_measurements = add_flipped_images(images, measurements)
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
keras_model = create_model()
keras_model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
keras_model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)  #20% validation
keras_model.save("model.h5")
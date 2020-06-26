# fix print format when using Pycharm terminal, has nothing to do with the code functionality
import ipykernel

import csv
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Dropout, Cropping2D, Conv2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from math import ceil
import random

examples = []

with open('./simple_data/driving_log_0.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        examples.append(line)

print('Data size is ', len(examples))

shuffle(examples)
train_samples, validation_samples = train_test_split(examples, test_size=0.2)


# source: https://github.com/ancabilloni/SDC-P3-BehavioralCloning/blob/master/model.py
def random_brightness(image):
    # Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand * hsv[:, :, 2]
    # Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img


def generator(samples_set, batch_size=32, val_flag=False):
    ShowME = True
    num_samples = len(samples_set)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples_set)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples_set[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = "./simple_data" + "/IMG/" + batch_sample[0].split('/')[-1]
                # 1
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                images.append(center_image)
                angles.append(center_angle)

                # 2
                center_image_flipped = np.fliplr(center_image)
                images.append(center_image_flipped)
                angles.append(-1.0 * center_angle)

                # Read left and right cameras images
                name = "./simple_data" + "/IMG/" + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                name = "./simple_data" + "/IMG/" + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)

                # Adjust the steering for left an right images as if they are centre camera images
                # correction = 0.15  # this is a parameter to tune
                steering_left = center_angle + 0.20
                steering_right = center_angle - 0.20

                # 3
                images.append(left_image)
                angles.append(steering_left)

                # 4
                images.append(right_image)
                angles.append(steering_right)

                # 5
                image_Brightnes = random_brightness(center_image)
                images.append(image_Brightnes)
                angles.append(center_angle)
                # 6
                image_Brightnes_flipped = np.fliplr(image_Brightnes)
                images.append(image_Brightnes_flipped)
                angles.append(-1.0 * center_angle)
                # 7
                right_image_flipped = np.fliplr(right_image)
                images.append(right_image_flipped)
                angles.append(-1.0 * steering_right)
                # 8
                left_image_flipped = np.fliplr(left_image)
                images.append(left_image_flipped)
                angles.append(-1.0 * steering_left)

            X_train = np.array(images)
            y_train = np.array(angles)
            if ShowME:
                print("\n Data length now is {} \n".format(X_train.shape))
                ShowME = False
            yield shuffle(X_train, y_train)


def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((75, 25), (0, 0)), input_shape=(3, 160, 320)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (2, 2), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


model = nVidiaModel()
model.summary()

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size, val_flag=False)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

filepath = "{epoch:02d}-{val_loss:.5f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

history = model.fit_generator(train_generator,
                              steps_per_epoch=ceil(len(train_samples) / batch_size),
                              validation_data=validation_generator,
                              validation_steps=ceil(len(validation_samples) / batch_size),
                              epochs=5, verbose=1, callbacks=[checkpoint])

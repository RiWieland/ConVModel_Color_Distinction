

import numpy as np
import pandas as pd
import os
import cv2

from random import randint

import matplotlib.pyplot as plt
from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense ,Conv2DTranspose, MaxPooling2D, concatenate, Dropout, Lambda, Flatten

from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D, Concatenate

def model():
    input_image = Input(shape=(112, 112, 3))

    # TOP BRANCH

    # first top convolution layer

    top_conv1 = Convolution2D(filters=48, kernel_size=(11, 11), strides=(4, 4),

                              input_shape=(224, 224, 3), activation='relu')(input_image)

    top_conv1 = BatchNormalization()(top_conv1)

    top_conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_conv1)

    # second top convolution layer

    # split feature map by half

    top_top_conv2 = Lambda(lambda x: x[:, :, :, :24])(top_conv1)

    top_bot_conv2 = Lambda(lambda x: x[:, :, :, 24:])(top_conv1)

    top_top_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv2)

    top_top_conv2 = BatchNormalization()(top_top_conv2)

    top_top_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_top_conv2)

    top_bot_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv2)

    top_bot_conv2 = BatchNormalization()(top_bot_conv2)

    top_bot_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_bot_conv2)

    # third top convolution layer

    # concat 2 feature map

    top_conv3 = Concatenate()([top_top_conv2, top_bot_conv2])

    top_conv3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(top_conv3)

    # fourth top convolution layer

    # split feature map by half

    top_top_conv4 = Lambda(lambda x: x[:, :, :, :96])(top_conv3)

    top_bot_conv4 = Lambda(lambda x: x[:, :, :, 96:])(top_conv3)

    top_top_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv4)

    top_bot_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv4)

    # fifth top convolution layer

    top_top_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv4)

    top_top_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_top_conv5)

    top_bot_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv4)

    top_bot_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_bot_conv5)


    # first bottom convolution layer

    bottom_conv1 = Convolution2D(filters=48, kernel_size=(11, 11), strides=(4, 4),

                                 input_shape=(224, 224, 3), activation='relu')(input_image)

    bottom_conv1 = BatchNormalization()(bottom_conv1)

    bottom_conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_conv1)

    # second bottom convolution layer

    # split feature map by half

    bottom_top_conv2 = Lambda(lambda x: x[:, :, :, :24])(bottom_conv1)

    bottom_bot_conv2 = Lambda(lambda x: x[:, :, :, 24:])(bottom_conv1)

    bottom_top_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv2)

    bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)

    bottom_top_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_top_conv2)

    bottom_bot_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv2)

    bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)

    bottom_bot_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_bot_conv2)

    # third bottom convolution layer

    # concat 2 feature map

    bottom_conv3 = Concatenate()([bottom_top_conv2, bottom_bot_conv2])

    bottom_conv3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_conv3)

    # fourth bottom convolution layer

    # split feature map by half

    bottom_top_conv4 = Lambda(lambda x: x[:, :, :, :96])(bottom_conv3)

    bottom_bot_conv4 = Lambda(lambda x: x[:, :, :, 96:])(bottom_conv3)

    bottom_top_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv4)

    bottom_bot_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv4)

    # fifth bottom convolution layer

    bottom_top_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv4)

    bottom_top_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_top_conv5)

    bottom_bot_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv4)

    bottom_bot_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_bot_conv5)

    # CONCATENATE

    conv_output = Concatenate()([top_top_conv5, top_bot_conv5, bottom_top_conv5, bottom_bot_conv5])

    # Flatten

    flatten = Flatten()(conv_output)

    # Fully-connected layer

    FC_1 = Dense(units=4096, activation='relu')(flatten)

    FC_1 = Dropout(0.6)(FC_1)

    FC_2 = Dense(units=4096, activation='relu')(FC_1)

    FC_2 = Dropout(0.6)(FC_2)

    output = Dense(units=4, activation='softmax')(FC_2)

    model = Model(inputs=input_image, outputs=output)

    return model


def model_test():
    inputs = Input(shape=(112, 112, 3))
    #s = Lambda(lambda x: x) (inputs)

    c1 = Conv2D(3, (11, 11), strides=4, activation='relu')(inputs)
    p1 = MaxPooling2D((3, 3), strides=2)(c1)
    N1 = BatchNormalization()(p1)

    c2a = Conv2D(64, (3, 3), strides=1, activation='relu')(N1)
    p2a = MaxPooling2D((3, 3), strides=2)(c2a)
    N2a = BatchNormalization()(p2a)

    c2b = Conv2D(64, (3, 3), strides=1, activation='relu')(N1)
    p2b = MaxPooling2D((3, 3), strides=2)(c2b)
    N2b = BatchNormalization()(p2b)

    c3a = Conv2D(96, (3,3), strides=1, activation='relu')(N2a)
    c3b = Conv2D(96, (3,3), strides=1, activation='relu')(N2b)

    c3 = concatenate([c3a, c3b])
    c3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(c3)
    flatten = Flatten()(c3)

    d1 = Dense(units=4096, activation='relu')(flatten)
    #d2 = Dropout(0.3)(d1)
    output = Dense(4, activation='softmax')(d1)
    model = Model(input=inputs, output=output)

    return model



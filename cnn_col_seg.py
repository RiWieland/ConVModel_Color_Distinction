

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



def model_OLD():
    inputs = Input((227, 227, 3))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.2) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(input=inputs, output=outputs)

    return model

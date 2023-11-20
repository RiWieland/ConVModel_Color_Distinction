#!/usr/bin/env python

# Difficulies:
# -- Efficient Data Pipeline to pass large tensors

# To Do:
# -- Check normalization (in original + aug dat): /255
# -- batch not consistent
# -- preformance differs with input size!?


import os as os
import time
import numpy as np

import tensorflow as tf
#from Archive.cnn_models import cnn_model_fn_complex
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam, Adamax
from datetime import datetime
import cv2
from keras.models import load_model
from random import randint


from cnn_col_seg import model
from dataprocessing import create_data_with_labels, init_datagen, generator_


train_dir = os.getcwd() + "/data/train_init/"
test_dir = os.getcwd() + "/data/test/"

X_train, y_train = create_data_with_labels(train_dir, 100)
X_val, y_val = create_data_with_labels(test_dir, 1000)

datagen = init_datagen()

#model = model_git()
model = model()
print(model.summary())

# batch 72
# Checkpoint_10_03_1135.h5
now = datetime.now()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
Model_Checkpoints = os.getcwd() + '/Save/' + 'Checkpoint_' + now.strftime("%d_%m_%H%M") + '.h5'

callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001, verbose=1)
        , ModelCheckpoint(Model_Checkpoints, verbose=1, save_best_only=True, save_weights_only=True)
        ]
batch = 80
#results = model.fit_generator(generator_(train_dir, batch, 'train'), validation_data=(X_val, y_val),steps_per_epoch=130, epochs=2, callbacks=callbacks)
results = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch), validation_data=(X_val, y_val), steps_per_epoch=120, epochs=1, callbacks=callbacks)

model_name = os.getcwd() + '/Save/' + 'model_' + now.strftime("%d_%m_%H%M") + '.h5'
model.save(model_name)
weights_name = os.getcwd() + '/Save/' + 'weights_' + now.strftime("%d_%m_%H%M") + '.h5'
model.save_weights(weights_name)



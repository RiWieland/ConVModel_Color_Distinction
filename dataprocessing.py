import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img


def create_data_with_labels(image_dir, batch):
    
    mug_dirs = [f for f in os.listdir(image_dir) if not f.startswith('.')]
    mug_files = []

    for mug_dir in mug_dirs:
        mug_image_files_ = [image_dir + mug_dir + '/' + '{0}'.format(f)
                           for f in os.listdir(image_dir + mug_dir) if not f.startswith('.')]

        mug_image_files = mug_image_files_[0:int(batch)]
        mug_files += [mug_image_files]

    num_images = len(mug_files[0])

    images_np_arr = np.empty([len(mug_files), num_images, 112, 112, 3], dtype=np.float32)

    for mug, _ in enumerate(mug_files):
        for mug_image in range(num_images):
            img = cv2.imread(mug_files[mug][mug_image])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)

            images_np_arr[mug][mug_image] = img / 255.

    data = images_np_arr[0]
    labels = np.full(num_images, int(mug_dirs[0][0]))

    for i in range(1, len(mug_dirs)):
    #for i in range(1, 500):
        data = np.append(data, images_np_arr[i], axis=0)
        labels = np.append(labels, np.full(num_images, int(mug_dirs[i][0])), axis=0)

    return data, to_categorical(labels, len(mug_dirs))

def generator_(image_dir, batch, mode):
    while True:

        if mode == 'train':
            features, labels = create_data_with_labels(image_dir, batch)

            yield features, labels

        if mode == 'eval':
            features, labels = create_data_with_labels(image_dir, batch)

            return features, labels

def init_datagen():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        #brightness_range=[0.2, 1.0],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    return datagen

def augment_data(image_arr):

    datagen = init_datagen()
    aug_rate = 36
    save_to_path = os.getcwd() + '/data/aug/'

    for counter in range(0, len(image_arr[0])):

        x = np.expand_dims(image_arr[counter], axis=0)

        aug = datagen.flow(x, save_to_dir=save_to_path, save_prefix='Keras', save_format='jpg')

    return aug



import csv
import cv2
import numpy as np
import re

np.random.seed(0)

# Load data csv file
def read_csv(file_name):
    lines = []
    with open(file_name) as driving_log:
        reader = csv.reader(driving_log)
        next(reader, None)
        for line in reader:
            lines.append(line)

    return lines


def load_image(image_path):
    pattern = re.compile(r'/|\\')
    file_name = pattern.split(image_path)[-1]
    current_path = 'data/IMG/' + file_name
    #print(current_path)
    image_bgr = cv2.imread(current_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb

def preprocess_data(lines):
    images = []
    steerings = []

    for line in lines:
        # centre
        images.append(load_image(line[0]))
        # left
        images.append(load_image(line[1]))
        # right
        images.append(load_image(line[2]))

        centre_steering = float(line[3])
        correction = 0.2
        # centre
        steerings.append(centre_steering)
        # left
        steerings.append(centre_steering+correction)
        # right
        steerings.append(centre_steering-correction)

    return images, steerings

def random_translate(image, steering, range_x=100, range_y=10):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering


def random_exposure(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_shadow(image, strength=0.50):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def augment_data(images, steerings):
    augmented_images = []
    augmented_steerings = []
    for image, steering in zip(images, steerings):
        # add original
        augmented_images.append(image)
        augmented_steerings.append(steering)
        # add horizontally flipped
        augmented_images.append(cv2.flip(image, 1))
        augmented_steerings.append(steering*-1.0)
        # add randomly translated
        image_augmented, steering_augmented = random_translate(image, steering)
        # add random exposure
        image_augmented = random_exposure(image_augmented)
        # add random shadow
        rand_shadow = np.random.uniform(0,1)
        if rand_shadow > 0.6:
            image_augmented = random_shadow(image_augmented)
        augmented_images.append(image_augmented)
        augmented_steerings.append(steering_augmented)

    return augmented_images, augmented_steerings


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping


def model_LeNet():
    model = Sequential()
    model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Conv2D(6, (5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def model_nvidia():
    model = Sequential()
    model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(24, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

import sklearn
import threading
from math import ceil
from random import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self
    
    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generator(samples, batch_size = 128):
    print('generator initialized')
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, steerings = preprocess_data(batch_samples)
            images, steerings = augment_data(images, steerings)
            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)

if __name__ == '__main__':

    print("Loading csv file ...")
    csv_file_name = 'data/driving_log.csv'
    lines = read_csv(csv_file_name)
    csv_file_name = 'data1/driving_log.csv'
    lines.extend(read_csv(csv_file_name))
    print("Finished loading csv file")

    print("Preprocessing images ...")
    '''
    images, steerings = preprocess_data(lines)
    images, steerings = augment_data(images, steerings)
    X_train = np.array(images)
    y_train = np.array(steerings)
    '''
    # This should be adjusted according to memory size
    batch_size = 64
    
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    print("Finished Preprocessing images")
    # Hyper parameters
    epochs_num = 10
    model = model_nvidia()
    model.summary()

    model.compile(loss='mse', optimizer='adam')
    checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4,
                                    verbose=1, mode='min')
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
    #                      callbacks=[checkpoint, early_stop], validation_split=0.2, shuffle=True)
    
    history_object = model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            callbacks=[checkpoint, early_stop],
            nb_epoch=epochs_num, verbose=1)
    
    # Plot the training and validation loss for each epoch
    print('Generating loss chart...')
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('model.png')

    print('Finished')
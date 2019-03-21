import csv
import cv2
import numpy as np


# Load data csv file
def read_csv(file_name):
    lines = []
    with open(file_name) as driving_log:
        reader = csv.reader(driving_log)
        next(reader, None)
        for line in reader:
            lines.append(line)

    return lines


def preprocess_data(lines):
    images = []
    steerings = []

    for line in lines:
        source_path = line[0]
        file_name = source_path.split('/')[-1]
        current_path = 'data/IMG/' + file_name
        image_bgr = cv2.imread(current_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)
        steering = float(line[3])
        steerings.append(steering)

    return images, steerings

def augment_data(images, steerings):
    augmented_images = []
    augmented_steerings = []
    for image, steering in zip(images, steerings):
        augmented_images.append(image)
        augmented_steerings.append(steering)
        augmented_images.append(cv2.flip(image, 1))
        augmented_steerings.append(steering*-1.0)

    return augmented_images, augmented_steerings


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping


def model_LeNet():
    model = Sequential()
    model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Convolution2D(6, (5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, (5,5), activation='relu'))
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
    model.add(Convolution2D(24, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

#model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Flatten())
#model.add(Dense(1))

if __name__ == '__main__':

    print("Loading csv file ...")
    csv_file_name = 'data/driving_log.csv'
    lines = read_csv(csv_file_name)
    print("Finished loading csv file")

    print("Preprocessing images ...")
    images, steerings = preprocess_data(lines)
    images, steerings = augment_data(images, steerings)
    X_train = np.array(images)
    y_train = np.array(steerings)
    print("Finished Preprocessing images")
    # Hyper parameters
    epochs = 5
    batch_size = 128
    model = model_nvidia()

    model.compile(loss='mse', optimizer='adam')
    checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4,
                                    verbose=1, mode='min')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                          callbacks=[checkpoint, early_stop], validation_split=0.2, shuffle=True)
    print('Finished')
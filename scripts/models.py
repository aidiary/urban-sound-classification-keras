from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=80, kernel_size=(57, 6), strides=(1, 1), padding='valid', input_shape=(60, 41, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
    model.add(Conv2D(filters=80, kernel_size=(1, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Flatten())
    model.add(Dense(units=5000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=5000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    return model

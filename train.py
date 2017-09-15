import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler


seed = 0
np.random.seed(seed)

if not os.path.exists('models'):
    os.mkdir('models')

train_features = np.load('data/train_features.npy')
train_labels = np.load('data/train_labels.npy')
test_features = np.load('data/test_features.npy')
test_labels = np.load('data/test_labels.npy')

# scale train/test data
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
scaler = scaler.fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

# create model
model = Sequential()
model.add(Dense(280, input_shape=(128, )))
model.add(Activation('relu'))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# callbacks
logger = CSVLogger('models/train.log')
weight_file = 'models/train.{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.h5'
checkpoint = ModelCheckpoint(weight_file,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')

# train
model.fit(train_features, train_labels,
          batch_size=32,
          epochs=128,
          verbose=1,
          validation_split=0.1,
          callbacks=[logger, checkpoint])

# evaluate

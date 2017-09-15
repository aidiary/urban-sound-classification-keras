import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler


if not os.path.exists('models'):
    os.mkdir('models')

train_features = np.load('data/train_features.npy')
train_labels = np.load('data/train_labels.npy')
test_features = np.load('data/test_features.npy')
test_labels = np.load('data/test_labels.npy')

print(np.mean(train_features), np.std(train_features))
print(np.mean(test_features), np.std(test_features))

# scale train/test data
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
scaler = scaler.fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

print(np.mean(train_features), np.std(train_features))
print(np.mean(test_features), np.std(test_features))

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
          validation_data=(test_features, test_labels),
          callbacks=[logger, checkpoint])

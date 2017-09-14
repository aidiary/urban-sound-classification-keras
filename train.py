from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint

# create model
model = Sequential()
model.add(Dense(n_hidden_units_1, input_shape=(n_dim, )))
model.add(Activation('relu'))
model.add(Dense(n_hidden_units_2))
model.add(Activation('relu'))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# callbacks
logger = CSVLogger('train.log')
weight_file = 'train.{epoch:02d}-{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(weight_file,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')

# train
model.fit(train_features, train_labels,
          batch_size=32,
          epochs=50,
          verbose=1,
          validation_data=(test_features, test_labels),
          callbacks=[logger, checkpoint])

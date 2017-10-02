import os
import librosa
import numpy as np
import pandas as pd
import keras
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from makedata import DATA_ROOT
from models import create_model


MODEL_DIR = '../models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


class Dataset:
    def __init__(self, features, fold_testing, fold_validation, shape):
        train = features[(features['fold'] != fold_testing) & (features['fold'] != fold_validation)]
        validation = features[(features['fold'] == fold_validation)]
        test = features[(features['fold'] == fold_testing)]
        print('train:', train.shape)
        print('validation:', validation.shape)
        print('test:', test.shape)

        self.shape = shape
        self.start = 'logspec_b0_f0'
        self.end = features.columns[-1]
        class_count = len(pd.unique(features['category']))

        X = train.loc[:, self.start:self.end].as_matrix()
        y = Dataset.to_one_hot(train['category'].as_matrix(), class_count)

        X_validation = validation.loc[:, self.start:self.end].as_matrix()
        y_validation = Dataset.to_one_hot(validation['category'].as_matrix(), class_count)

        X_test = test.loc[:, self.start:self.end].as_matrix()
        y_test = Dataset.to_one_hot(test['category'].as_matrix(), class_count)

        # メルスペクトログラムのdBは平均0、標準偏差1に正規化
        X_mean = np.mean(X)
        X_std = np.std(X)

        # 訓練データの (mean, std) でバリデーションとテストデータも正規化
        X = (X - X_mean) / X_std
        X_validation = (X_validation - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        X = np.reshape(X, self.shape, order='F')
        X_validation = np.reshape(X_validation, self.shape, order='F')
        X_test = np.reshape(X_test, self.shape, order='F')

        # generate delta
        X = self.generate_deltas(X)
        X_validation = self.generate_deltas(X_validation)
        X_test = self.generate_deltas(X_test)

        self.X, self.y = X, y
        self.X_validation, self.y_validation = X_validation, y_validation
        self.X_test, self.y_test = X_test, y_test


    def generate_deltas(self, X):
        new_dim = np.zeros(np.shape(X))
        X = np.concatenate((X, new_dim), axis=3)  # 3 = channel
        del new_dim

        for i in range(len(X)):
            X[i, :, :, 1] = librosa.feature.delta(X[i, :, :, 0])

        return X

    @classmethod
    def to_one_hot(cls, labels, class_count):
        one_hot_enc = np.zeros((len(labels), class_count))
        for r in range(len(labels)):
            one_hot_enc[r, labels[r]] = 1
        return one_hot_enc


def main():
    # load data
    urban_features = pd.read_pickle(os.path.join(DATA_ROOT, 'urban_features.pkl'))
    dataset = Dataset(urban_features, fold_testing=1, fold_validation=10, shape=(-1, 60, 41, 1))
    print(dataset.X.shape, dataset.y.shape)
    print(dataset.X_validation.shape, dataset.y_validation.shape)
    print(dataset.X_test.shape, dataset.y_test.shape)

    model = create_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), #SGD(lr=0.002, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, 'model.{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.h5'),
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='auto')
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/1')

    epochs = 100
    batch_size = 128

    model.fit(dataset.X, dataset.y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(dataset.X_validation, dataset.y_validation),
              callbacks=[checkpoint, tensorboard])


if __name__ == '__main__':
    main()

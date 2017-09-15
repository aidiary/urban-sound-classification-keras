import glob
import os
import librosa
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def extract_feature(file_name):
    y, sr = librosa.load(file_name)

    # melspectrogram (128 dim.)
    melgram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)

    return melgram


def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):
    features = np.empty((0, 128))
    labels = []
    filenames = []

    for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print('***', sub_dir, fn)
            try:
                melgram = extract_feature(fn)
            except Exception as e:
                print('Error encountered while parsing file: ', fn)
                continue

            features = np.vstack((features, melgram))

            # data/UrbanSound8K/audio/fold1/101415-3-0-2.wav => 3
            labels.append(os.path.basename(fn).split('-')[1])
            filenames.append(fn)

    return features, np.array(labels, dtype=np.int), np.array(filenames)


def main():
    parent_dir = './data/sample'
    train_sub_dirs = ['train']
    test_sub_dirs = ['test']

    # parent_dir = './data/UrbanSound8k/audio/'
    # train_sub_dirs = ['fold1', 'fold2']
    # test_sub_dirs = ['fold3']

    # extract features from audio files
    train_features, train_labels, train_filenames = parse_audio_files(parent_dir, train_sub_dirs)
    test_features, test_labels, test_filenames = parse_audio_files(parent_dir, test_sub_dirs)

    # convert to one hot encode
    train_labels = train_labels.reshape(len(train_labels), 1)
    test_labels = test_labels.reshape(len(test_labels), 1)

    enc = OneHotEncoder(sparse=False)
    train_labels = enc.fit_transform(train_labels)
    test_labels = enc.fit_transform(test_labels)

    # save to file
    np.save('data/train_features.npy', train_features)
    np.save('data/train_labels.npy', train_labels)
    np.save('data/train_filenames.npy', train_filenames)

    np.save('data/test_features.npy', test_features)
    np.save('data/test_labels.npy', test_labels)
    np.save('data/test_filenames.npy', test_filenames)


if __name__ == '__main__':
    main()

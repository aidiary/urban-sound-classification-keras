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


def parse_audio_files(target):
    features = np.empty((0, 128))
    labels = []
    filenames = []

    print()
    for fn in glob.glob(target):
        print('***', fn)
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
    target = './data/UrbanSound8K/audio/*/*.wav'

    # extract features from audio files
    features, labels, filenames = parse_audio_files(target)

    assert len(features) != 0
    assert len(features) == len(labels) == len(filenames)

    # convert to one hot encode
    labels = labels.reshape(len(labels), 1)

    enc = OneHotEncoder(sparse=False)
    labels = enc.fit_transform(labels)

    # save to file
    np.save('data/features.npy', features)
    np.save('data/labels.npy', labels)
    np.save('data/filenames.npy', filenames)


if __name__ == '__main__':
    main()

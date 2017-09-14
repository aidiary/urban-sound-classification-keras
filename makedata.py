import glob
import os
import librosa
import numpy as np


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))

    # MFCC 40 dim.
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40), axis=1)

    # chromagram 12 dim.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate), axis=1)

    # melspectrogram 128 dim.
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate), axis=1)

    # spectral contrast 7 dim.
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate), axis=1)

    # tonal centroid features 6 dim.
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate), axis=1)

    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):
    features, labels = np.empty((0, 193)), np.empty(0)
    for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print('***', sub_dir, fn)
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
                print('Error encountered while parsing file: ', fn)
                continue
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            # data/UrbanSound8K/audio/fold1/101415-3-0-2.wav => 3
            labels = np.append(labels, fn.split('/')[-1].split('-')[1])
    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encodes(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def main():
    parent_dir = './data/UrbanSound8K/audio'
    train_sub_dirs = ['fold1', 'fold2']
    test_sub_dirs = ['fold3']

    train_features, train_labels = parse_audio_files(parent_dir, train_sub_dirs)
    test_features, test_labels = parse_audio_files(parent_dir, test_sub_dirs)

    train_labels = one_hot_encodes(train_labels)
    test_labels = one_hot_encodes(test_labels)

    print(train_features.shape, train_labels.shape)
    print(test_features.shape, test_labels.shape)

    np.save('data/train_features.npy', train_features)
    np.save('data/train_labels.npy', train_labels)
    np.save('data/test_features.npy', test_features)
    np.save('data/test_labels.npy', test_labels)


if __name__ == '__main__':
    main()

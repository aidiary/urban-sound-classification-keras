import os
import glob
import argparse
import librosa
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def extract_feature(file_name, n_mels=128, n_fft=2048):
    y, sr = librosa.load(file_name)

    # melspectrogram (n_mels, n_frames)
    melgram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft)

    # (n_mels, )
    melgram = np.mean(melgram, axis=1)

    return melgram


def parse_audio_files(parent_dir, sub_dirs, n_mels=128, n_fft=2048):
    features = np.empty((0, n_mels))
    labels = []
    filenames = []

    for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, '*.wav')):
            print('***', sub_dir, fn)
            try:
                melgram = extract_feature(fn, n_mels, n_fft)
            except Exception as e:
                print('Error encountered while parsing file: ', fn, e)
                continue

            features = np.vstack((features, melgram))

            # data/UrbanSound8K/audio/fold1/101415-3-0-2.wav => 3
            labels.append(os.path.basename(fn).split('-')[1])
            filenames.append(fn)

    return features, np.array(labels, dtype=np.int), np.array(filenames)


def main():
    parser = argparse.ArgumentParser(description='make data for UrbanSound8K')
    parser.add_argument('n_mels', type=int, default=128, help='number of Mel bins')
    parser.add_argument('n_fft', type=int, default=2048, help='length of the FFT window')
    parser.add_argument('out_dir', type=str, help='output directory')
    args = parser.parse_args()

    # parent_dir = './data/sample'
    # train_sub_dirs = ['train']
    # test_sub_dirs = ['test']

    parent_dir = './data/UrbanSound8K/audio/'
    train_sub_dirs = ['fold%d' % i for i in range(1, 10)]
    test_sub_dirs = ['fold10']

    # extract features from audio files
    train_features, train_labels, train_filenames = parse_audio_files(
        parent_dir, train_sub_dirs, args.n_mels, args.n_fft)

    test_features, test_labels, test_filenames = parse_audio_files(
        parent_dir, test_sub_dirs, args.n_mels, args.n_fft)

    # convert to one hot encode
    train_labels = train_labels.reshape(len(train_labels), 1)
    test_labels = test_labels.reshape(len(test_labels), 1)

    enc = OneHotEncoder(sparse=False)
    train_labels = enc.fit_transform(train_labels)
    test_labels = enc.fit_transform(test_labels)

    # save to file
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    np.save(os.path.join(args.out_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(args.out_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(args.out_dir, 'train_filenames.npy'), train_filenames)

    np.save(os.path.join(args.out_dir, 'test_features.npy'), test_features)
    np.save(os.path.join(args.out_dir, 'test_labels.npy'), test_labels)
    np.save(os.path.join(args.out_dir, 'test_filenames.npy'), test_filenames)


if __name__ == '__main__':
    main()

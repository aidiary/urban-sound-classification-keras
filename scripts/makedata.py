import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydub
import librosa

import multiprocessing
from joblib import Parallel, delayed


DATA_ROOT = '../data'
CPU_COUNT = multiprocessing.cpu_count()


def load_audio(path, duration):
    # duration (ms)
    audio = pydub.AudioSegment.silent(duration=duration)
    try:
        audio = audio.overlay(pydub.AudioSegment.from_file(path).set_frame_rate(22050).set_channels(1))[:duration]
    except:
        return None
    # -32768 - 32767
    raw = np.fromstring(audio._data, dtype='int16')
    # -1 - +1
    raw = (raw + 0.5) / (32767 + 0.5)
    return raw


def load_urbansound():
    """Load raw audio and metadata from the UrbanSound8K dataset."""
    if os.path.isfile(os.path.join(DATA_ROOT, 'urban_meta.pkl')) and os.path.isfile(os.path.join(DATA_ROOT, 'urban_audio.npy')):
        rows_meta = pd.read_pickle(os.path.join(DATA_ROOT, 'urban_meta.pkl'))
        rows_audio = np.load(os.path.join(DATA_ROOT, 'urban_audio.npy'))
        return rows_meta, rows_audio

    metadata = pd.read_csv(os.path.join(DATA_ROOT, 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv'))

    b = 0
    batch_size = 1000
    rows_meta = []
    rows_audio = []
    while len(metadata[b * batch_size:(b + 1) * batch_size]):
        for key, row in metadata[b * batch_size:(b + 1) * batch_size].iterrows():
            filename = row['slice_file_name']
            fold = row['fold']
            category = row['classID']
            category_name = row['class']
            rows_meta.append(pd.DataFrame({'filename':filename,
                                           'fold':fold,
                                           'category':category,
                                           'category_name':category_name}, index=[0]))
            audio_path = os.path.join(DATA_ROOT, 'UrbanSound8K', 'audio', 'fold%d' % fold, filename)
            audio = load_audio(audio_path, 4000)
            if audio is not None:
                rows_audio.append(load_audio(audio_path, 4000))
        b = b + 1
        # この2行必要？最後に1回だけやればよいのでは？
        rows_meta = [pd.concat(rows_meta, ignore_index=True)]
        rows_audio = [np.vstack(rows_audio)]

        # for debug
        print('Loaded batch {} ({} / {})'.format(b, b * batch_size, len(metadata)))

    rows_meta = rows_meta[0]
    rows_audo = rows_audio[0]
    rows_meta[['category', 'fold']] = rows_meta[['category', 'fold']].astype(int)

    # save to files
    rows_meta.to_pickle(os.path.join(DATA_ROOT, 'urban_meta.pkl'))
    np.save(os.path.join(DATA_ROOT, 'urban_audio.npy'), rows_audio)

    return rows_meta, rows_audio


def extract_segments(clip, filename, fold, category, category_name, frames):
    FRAMES_PER_SEGMENT = frames - 1  # 41 frames ~= 950 ms
    WINDOW_SIZE = 512 * FRAMES_PER_SEGMENT  # 23 ms per frame
    STEP_SIZE = 512 * FRAMES_PER_SEGMENT // 2  # 512 * 20 = 10240
    BANDS = 60

    s = 0
    segments = []

    normalization_factor = 1 / np.max(np.abs(clip))
    clip = clip * normalization_factor

    while len(clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]) == WINDOW_SIZE:
        signal = clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]

        melspec = librosa.feature.melspectrogram(signal, sr=22050, n_fft=1024, hop_length=512, n_mels=BANDS)
        logspec = librosa.logamplitude(melspec)  # to dB
        # (60, 41) => (41, 60) => (2460, 1) => (1, 2460)
        logspec = logspec.T.flatten()[:, np.newaxis].T
        # スペクトログラムが1次元arrayになってしまったのでどのフレームのどのバンドか名前をつけておく
        logspec = pd.DataFrame(data=logspec, dtype='float32', index=[0],
                               columns=list('logspec_b{}_f{}'.format(i % BANDS, i // BANDS) for i in range(np.shape(logspec)[1])))
        # 無音のフレームでなければ処理対象とする
        if np.mean(logspec.as_matrix()) > -70.0:
            segment_meta = pd.DataFrame({
                    'filename': filename,
                    'fold': fold,
                    'category': category,
                    'category_name': category_name,
                    's_begin': s * STEP_SIZE,
                    's_end': s * STEP_SIZE + WINDOW_SIZE}, index=[0])
            segments.append(pd.concat((segment_meta, logspec), axis=1))
        s = s + 1

    segments = pd.concat(segments, ignore_index=True)
    return segments


def extract_features(meta, audio, frames=41):
    np.random.seed(20170927)
    batch_size = 100
    segments = []
    # 各バッチ（startからend）の各音声データから並列に特徴抽出
    for b in range(len(audio) // batch_size + 1):
        start = b * batch_size
        end = (b + 1) * batch_size
        if end > len(audio):
            end = len(audio)
        segments.extend(Parallel(n_jobs=CPU_COUNT)(delayed(extract_segments)(
                        audio[i, :],
                        meta.loc[i, 'filename'],
                        meta.loc[i, 'fold'],
                        meta.loc[i, 'category'],
                        meta.loc[i, 'category_name'],
                        frames) for i in range(start, end)))
        segments = [pd.concat(segments, ignore_index=True)]

        # for debug
        print('{} / {}'.format(end, len(audio)))
    return segments[0]


def main():
    urban_meta, urban_audio = load_urbansound()
    if os.path.isfile(os.path.join(DATA_ROOT, 'urban_features.pkl')):
        urban_features = pd.read_pickle(os.path.join(DATA_ROOT, 'urban_features.pkl'))
    else:
        urban_features = extract_features(urban_meta, urban_audio)
        urban_features.to_pickle(os.path.join(DATA_ROOT, 'urban_features.pkl'))


if __name__ == '__main__':
    main()

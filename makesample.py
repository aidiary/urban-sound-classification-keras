import os
import glob
import random
import shutil

traindir = './data/sample/train'
testdir = './data/sample/test'

if not os.path.exists(traindir):
    os.makedirs(traindir)

if not os.path.exists(testdir):
    os.makedirs(testdir)

# copy train samples for each class
for i in range(0, 10):
    trains = glob.glob('./data/UrbanSound8k/audio/fold1/*-%d-*-*.wav' % i)
    random.shuffle(trains)
    for j in range(3):
        print(trains[j])
        shutil.copy(trains[j], traindir)

# copy test samples for each class
for i in range(0, 10):
    tests = glob.glob('./data/UrbanSound8k/audio/fold3/*-%d-*-*.wav' % i)
    random.shuffle(tests)
    for j in range(2):
        print(tests[j])
        shutil.copy(tests[j], testdir)

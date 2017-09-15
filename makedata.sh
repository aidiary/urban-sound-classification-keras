#!/bin/bash
python makedata.py 128 2048 data/mel-128-2048
python makedata.py 256 2048 data/mel-256-2048

python makedata.py 128 1024 data/mel-128-1024
python makedata.py 128 4096 data/mel-128-4096

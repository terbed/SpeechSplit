import os
import sys
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT

from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
from hparams import hparams


mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

gender = 'M'
wav_path = ''

if gender == 'M':
    lo, hi = 50, 250
elif gender == 'F':
    lo, hi = 100, 600
else:
    raise ValueError

prng = RandomState(10)
x, fs = sf.read(os.path.join(wav_path))
assert fs == 16000
if x.shape[0] % 256 == 0:
    x = np.concatenate((x, np.array([1e-06])), axis=0)
y = signal.filtfilt(b, a, x)
wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

# compute spectrogram
D = pySTFT(wav).T
D_mel = np.dot(D, mel_basis)
D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
S = (D_db + 100) / 100

# extract f0
f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, fs, 256, min=lo, max=hi, otype=2)
index_nonzero = (f0_rapt != -1e10)
mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

assert len(S) == len(f0_rapt)




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
from utils import pad_seq_to_2
from utils import speaker_normalization
from utils import pySTFT
import librosa as lr

from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
from hparams import hparams

from synthesis import build_model
from synthesis import wavegen

import torch
tr = torch

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")
print(device)
G = Generator(hparams).eval().to(device)
g_checkpoint = torch.load('assets/660000-G.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])

P = F0_Converter(hparams).eval().to(device)
p_checkpoint = torch.load('assets/640000-P.ckpt', map_location=lambda storage, loc: storage)
P.load_state_dict(p_checkpoint['model'])

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

gender = 'M'
# wav_path = '/home/deep06/DYS/DATA/zzz/dys/000.wav'
wav_path = '/home/deep06/DYS/DATA/zzz/ctrl/0-000.wav'

if gender == 'M':
    lo, hi = 50, 250
elif gender == 'F':
    lo, hi = 100, 600
else:
    raise ValueError

prng = RandomState(10)
# x, fs = sf.read(os.path.join(wav_path))
x, fs  = lr.load(os.path.join(wav_path), sr=16000)
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

# S:utterance spectrogram, f0_norm:normalized pitch contour
f0_quantized = quantize_f0_numpy(f0_norm)[0]
f0_onehot = f0_quantized[np.newaxis, :, :]
print(f0_onehot.shape)
f0_onehot = f0_onehot[:,:192,:]
f0_onehot, _ = pad_seq_to_2(f0_onehot, 192)
f0_onehot = torch.from_numpy(f0_onehot).to(device)

# concat pitch contour to freq axis (cols)
S = S[np.newaxis,:192,:]
S, _ = pad_seq_to_2(S, 192)
uttr = torch.from_numpy(S.astype(np.float32)).to(device)

f0_onehot = tr.zeros_like(f0_onehot)
uttr_f0 = torch.cat((uttr, f0_onehot), dim=-1)

# Generate back from components
emb = tr.zeros(1, 82).to(device)
print(uttr_f0.shape, uttr.shape, emb.shape)

# uttr_f0 = tr.zeros_like(uttr_f0)
out = G(uttr_f0, uttr, emb)

# Synthesize wav back
model = build_model().to(device)
checkpoint = torch.load("assets/checkpoint_step001000000_ema.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])

print(out.shape)
waveform = wavegen(model, c=out.squeeze().cpu())
# librosa.output.write_wav('results/'+name+'.wav', waveform, sr=16000)
sf.write('results/back_synthesized-zeros-pitch.wav', waveform, 16000, subtype='PCM_24')


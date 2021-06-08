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
from model import GetCodes
from hparams import hparams

from synthesis import build_model
from synthesis import wavegen

import pickle
import os

import torch
tr = torch

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")
print(device)
encoders = GetCodes(hparams).eval().to(device)
g_checkpoint = torch.load('assets/660000-G.ckpt', map_location=lambda storage, loc: storage)
encoders.load_state_dict(g_checkpoint['model'])
print("Succesfully loaded")

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

# Get the codes for each utterance and save the corresponding codes in a large pickle
# List(Dict{'dys', 'ctrl'}) - numpy ndarray in the value part of the dict

src_path = "/home/terbed/PROJECTS/DYS/DATA/UAS-subset/M05"
trg_path = "/home/terbed/PROJECTS/DYS/DATA/UAS-subset/CM10"
lo, hi = 50, 250

src_speaker = src_path.split("/")[-1]
trg_speaker = trg_path.split("/")[-1]

_, _, fnames = next(os.walk(src_path))
database = []


def load_wav(wav_path):
    prng = RandomState(10)
    # x, fs = sf.read(os.path.join(wav_path))
    x, fs = lr.load(os.path.join(wav_path), sr=16000)
    x, _ = lr.effects.trim(x, top_db=15)
    assert fs == 16000
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
    return wav


def compute_spect(wav):
    # compute spectrogram
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100

    S = S[np.newaxis, :, :]
    if S.shape[1] <= 192:
        S, _ = pad_seq_to_2(S, 192)
    uttr = torch.from_numpy(S.astype(np.float32)).to(device)

    return uttr


def extract_f0(wav, fs):
    f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

    f0_quantized = quantize_f0_numpy(f0_norm)[0]
    f0_onehot = f0_quantized[np.newaxis, :, :]
    print(f0_onehot.shape)

    if f0_onehot.shape[1] <= 192:
        f0_onehot, _ = pad_seq_to_2(f0_onehot, 192)

    return torch.from_numpy(f0_onehot).to(device)


def get_codes(wav):
    with torch.no_grad():
        uttr = compute_spect(wav)
        f0 = extract_f0(wav, 16000)
        uttr_f0 = torch.cat((uttr, f0), dim=-1)
        speaker_emb = tr.zeros(1, 82).to(device)

        assert uttr.shape[1] == f0.shape[1], "S and f must be same size..."
        if uttr.shape[1] == 192:
            codes = encoders(uttr_f0, uttr, speaker_emb)
            return codes
        else:
            return None


for name in fnames:
    print(f"Working on {name}...")
    current_utter_code1 = name.split("_")[1]
    current_utter_code2 = name.split("_")[1]

    trg_name = name.split("_")
    trg_name[0] = trg_speaker
    trg_name = "_".join(trg_name)

    wav = load_wav(os.path.join(src_path, name))
    src_codes = get_codes(wav)

    wav = load_wav(os.path.join(trg_path, trg_name))
    trg_codes = get_codes(wav)

    if src_codes is not None and trg_codes is not None:
        trg_codes = trg_codes.cpu().numpy()
        src_codes = src_codes.cpu().numpy()
        assert src_codes.shape == trg_codes.shape, "The size of the codes must equal!"
        database.append({"src": src_codes, "trg": trg_codes})

# Store data (serialize)
with open('dbase.pkl', 'wb') as handle:
    pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)

# read pickle:
# with open("dbase.pkl", "rb") as handle:
#     d = pickle.load(handle)

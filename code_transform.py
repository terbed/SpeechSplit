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

import torch
tr = torch

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")
print(device)
encoders = GetCodes(hparams).eval().to(device)
g_checkpoint = torch.load('assets/660000-G.ckpt', map_location=lambda storage, loc: storage)
encoders.load_state_dict(g_checkpoint['model'])
print("Succesfully loaded")

# Get the codes for each utterance and save the corresponding codes in a large pickle
# List(Dict{'dys', 'ctrl'}) - numpy ndarray in the value part of the dict

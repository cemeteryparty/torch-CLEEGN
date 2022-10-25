import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keras_CLEEGN
from keras_CLEEGN.utils.model_tracer import EEG_Viewer
from keras_CLEEGN.utils.process_eeg import ar_through_cleegn
from keras_CLEEGN.utils.process_eeg import segment_eeg
from keras_CLEEGN.models.cleegn import CLEEGN

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy
import json
import math
import time
import mne

BCI_CHALL_EVENT = "/mnt/left/phlai_DATA/bci-challenge-event/"
x_basepath = "/mnt/left/phlai_DATA/bci-challenge/original"
y_basepath = "/mnt/left/lambert/ner2015_together/fulldata/onlybrain"
x_fpath = os.path.join(x_basepath, "Data_S{}.set")
y_fpath = os.path.join(y_basepath, "Data_S{}_Sess.set")

def Ep_mse(model, x_content, y_content, sfreq, events):
    stride = math.ceil(1.0 * sfreq)
    tmin, tmax = math.ceil(0.25 * sfreq), math.ceil(1.25 * sfreq)
    rdelta = max(tmin, tmax) * 2

    mse = losses.MeanSquaredError()
    x_epochs, y_epochs, p_epochs = [], [], []
    for event in events:
        centr = event[0]
        y = ar_through_cleegn(
            x_content[:, centr - rdelta: centr + rdelta],
            model, math.ceil(1.0 * sfreq)
        )
        x_epochs.append(x_content[:, centr - tmin: centr + tmax])
        y_epochs.append(y_content[:, centr - tmin: centr + tmax])
        p_epochs.append(y[:, rdelta - tmin: rdelta + tmax])
    x_epochs = np.array(x_epochs)
    y_epochs = np.array(y_epochs)
    p_epochs = np.array(p_epochs)
    return [
        mse(x_epochs, y_epochs).numpy(),
        mse(x_epochs, p_epochs).numpy(),
        mse(y_epochs, p_epochs).numpy()
    ]

NUM_SBJS = 12
SAVE_PATH = f"tmpfile/SbjsDiv/{NUM_SBJS}"
NUM_EPOCHS  = 30 #cfg["epochs"]
BATCH_SIZE  = 32
PREFIX = f"{NUM_SBJS}sbj"

n_trial = 1
ginfo = ["06", "13", "18", "14", "22", "26", "24", "02", "21", "12", "20", "23", "07", "16", "17", "11"]
tra_sbj = ["06", "13", "18", "14", "21", "12", "20", "23", "07", "16", "17", "11"]
tst_sbj = ["22", "26", "24", "02"]
for si in range(0, len(tra_sbj), NUM_SBJS):
    print(f"trials-{n_trial}")
    sbj_fold = []
    while len(sbj_fold) != NUM_SBJS:
        sbj_fold.append(tra_sbj[si])
        si += 1
    
    for sbj in tst_sbj:
        x_raw = mne.io.read_raw_eeglab(x_fpath.format(sbj), verbose=0)
        y_raw = mne.io.read_raw_eeglab(y_fpath.format(sbj), verbose=0)
        x_content = x_raw.get_data() * 1e6
        y_content = y_raw.get_data() * 1e6
        sfreq = x_raw.info["sfreq"]
        
        rstep = math.ceil(1.0 * sfreq) # sec
        model_path = os.path.join(SAVE_PATH, f"{PREFIX}-{n_trial}.h5")
        model = models.load_model(model_path)
        events = scipy.io.loadmat(os.path.join(BCI_CHALL_EVENT, f"Data_S{sbj}_event.mat"))["events"]
        epmse = Ep_mse(model, x_content, y_content, sfreq, events)
        print(f"{sbj}:", epmse)
    print()
    n_trial += 1

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keras_CLEEGN
from keras_CLEEGN.utils.process_eeg import ar_through_cleegn
from keras_CLEEGN.klosses import RootMeanSquaredError

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import losses

from scipy.io import loadmat
import numpy as np
import json
import math
import time
import mne
import csv

BCI_CHALL_EVENT = "/mnt/left/phlai_DATA/bci-challenge-event/"

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

def read_json(filepath):
    if filepath is None:
        return None
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content 

trial_name = "bc-56ch_asr32-ica"
ginfo = ["06", "13", "18", "14"] + ["22", "26", "24", "02"] + ["21", "12", "20", "23"] + ["07", "16", "17", "11"]
# basepath = "/mnt/left/home/2022/phlai410277/bci-challenge"
basepath = "/mnt/left/phlai_DATA/keras-CLEEGN/tmpfile"
basepath = os.path.join(basepath, trial_name)

filename = os.path.join(
    os.path.dirname(__file__), f"0_csv/{trial_name}.csv"
)
fd = open(filename, "w")
csv_writer = csv.writer(fd)
csv_writer.writerow(["Sbj","model","Ep-MSE(x,GT)","Ep-MSE(x,C)","Ep-MSE(GT,C)"])
sbjs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
for sid in sbjs:
    gid = ginfo.index(sid) // 4 + 1
    anno_path = os.path.join(basepath, "set_{}/set_test.json".format(gid))
    conf_path = os.path.join(basepath, "set_{}/config.json".format(gid))
    anno = read_json(anno_path)
    cfg = read_json(conf_path)
    assert cfg["duration-unit"] in ["sample-point", "sec"]
    sfreq     = anno["sfreq"]
    picks     = anno["ch_names"]

    # time seq center, +-duration
    events = loadmat(os.path.join(BCI_CHALL_EVENT, f"Data_S{sid}_event.mat"))["events"]

    x_fpath = os.path.join(anno["x_basepath"], anno["x_fpath"])
    y_fpath = os.path.join(anno["y_basepath"], anno["y_fpath"])
    x_raw = mne.io.read_raw_eeglab(x_fpath.format(sid), verbose=0)
    y_raw = mne.io.read_raw_eeglab(y_fpath.format(sid), verbose=0)
    x_content = x_raw.get_data() * 1e6
    y_content = y_raw.get_data() * 1e6

    if picks is not None:
        pick_idx = [x_raw.ch_names.index(ch_name) for ch_name in picks]
        x_content = x_content[pick_idx]
        y_content = y_content[pick_idx]
    else:
        picks = x_raw.ch_names

    rstep = math.ceil(1.0 * sfreq) # sec

    mtype = ["_tra", "_val"]
    for mt in mtype[1:]:
        model_name = cfg["model_name"] + mt
        model_path = os.path.join(basepath, "set_{}/{}.h5".format(gid, model_name))
        try:
            model = models.load_model(model_path)
        except ValueError:
            custom_objs = {"RootMeanSquaredError": RootMeanSquaredError}
            model = models.load_model(model_path, custom_objects=custom_objs)
        epmse = Ep_mse(model, x_content, y_content, sfreq, events)
        csv_writer.writerow([f"S{sid}", model_name] + epmse)
fd.close()

""" Old rough class
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keras_CLEEGN
from keras_CLEEGN.utils.process_eeg import ar_through_cleegn
from keras_CLEEGN.models.EEGModels import SCCNet
from keras_CLEEGN.models.EEGModels import EEGNet

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.io import loadmat
import numpy as np
import json
import math
import time
import mne
import csv

BCI_CHALL_EVENT = "/mnt/left/phlai_DATA/bci-challenge-event/"

class MyModelCkpt(callbacks.Callback):
    def __init__(self, basepath, monitor="val_acc", mode="max"):
        super().__init__()
        self.basepath = basepath
        self.mode = mode
        self.monitor = monitor
        self.bound = np.inf if mode == "min" else (-np.inf)

    def on_epoch_end(self, epoch, logs=None):
        metric = logs[self.monitor]
        if (self.mode == "min" and metric < self.bound) or (self.mode == "max" and metric > self.bound):
            self.bound = metric
            self.model.save(os.path.join(self.basepath, "sv_weight.h5py"))
        # print()

def read_json(filepath):
    if filepath is None:
        return None
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content 

basepath = "/mnt/left/home/2022/phlai410277/" # os.path.expanduser("~")
basepath = os.path.join(basepath, "bci-challenge/bc-56ch_ica")
ginfo = ["06", "13", "18", "14"] + ["22", "26", "24", "02"] + ["21", "12", "20", "23"] + ["07", "16", "17", "11"]
status = "pred" # "noisy", "clean", "pred"

filename = os.path.join(
    os.path.dirname(__file__), "bc-56ch_onlybrain_cleegn.csv"
)
fd = open(filename, "a")
csv_writer = csv.writer(fd)
csv_writer.writerow(
    ["sbj_id", "trial", "0fs", "1fs", "auc", "00", "01", "10", "11"]
)
sbjs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
if sys.argv[1] in sbjs:
    sbjs = [sys.argv[1]]
for sid in sbjs:
    print(f"Trial on Subject-S{sid}")
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
    
    """ default: ground truth """
    fpath = os.path.join(anno["y_basepath"], anno["y_fpath"])
    if status in ["noisy", "pred"]:
        fpath = os.path.join(anno["x_basepath"], anno["x_fpath"])
    mne_raw = mne.io.read_raw_eeglab(fpath.format(sid))
    mne.rename_channels(mne_raw.info, mapping={"P08": "PO8"})
    eeg_data = mne_raw.get_data() * 1e6

    if picks is not None:
        pick_idx = [mne_raw.ch_names.index(ch_name) for ch_name in picks]
        eeg_data = eeg_data[pick_idx]
    else:
        picks = mne_raw.ch_names

    epochs, labels = [], []
    tmin, tmax = math.ceil(0.25 * sfreq), math.ceil(1.25 * sfreq)
    rdelta = max(tmin, tmax) * 2
    for event_id in range(events.shape[0]):
        centr = events[event_id][0]
        labels.append(events[event_id][2])
        epoch = None
        if status == "pred":
            model_pfx = cfg["model_name"]
            model_path = os.path.join(basepath, f"set_{gid}/{model_pfx}_val.h5")
            cleegn = models.load_model(model_path)
            epoch = ar_through_cleegn(
                eeg_data[:, centr - rdelta: centr + rdelta],
                cleegn, math.ceil(1.0 * sfreq)
            )[:, rdelta - tmin: rdelta + tmax]
        else:
            epoch = eeg_data[:, centr - tmin: centr + tmax]
        epochs.append(epoch)
    epochs = np.array(epochs)
    labels = np.array(labels)

    for it in range(20):
        print(f"Trial on Subject-S{sid}-{it + 1}")
        c0 = [i for i in range(labels.size) if labels[i] == 0]
        c1 = [i for i in range(labels.size) if labels[i] == 1]
        c0_tra, c0 = train_test_split(c0, test_size=0.6)
        c1_tra, c1 = train_test_split(c1, test_size=0.6)
        c0_val, c0_tst = train_test_split(c0, test_size=0.5)
        c1_val, c1_tst = train_test_split(c1, test_size=0.5)
        c_tra = c0_tra + c1_tra
        c_val = c0_val + c1_val
        c_tst = c0_tst + c1_tst

        epochs_ = np.expand_dims(epochs, axis=3)
        labels_ = utils.to_categorical(labels, 2)
        x_train, y_train = epochs_[c_tra], labels_[c_tra]
        x_valid, y_valid = epochs_[c_val], labels_[c_val]
        x_test, y_test = epochs_[c_tst], labels_[c_tst]
        # print("train:", x_train.shape, y_train.shape)
        # print("test:", x_test.shape, y_test.shape)
        # print("valid:", x_valid.shape, y_valid.shape)
        # exit(0)
        """
        def SCCNet(nb_classes, Chans=64, Samples=128, fs=128, dropoutRate=0.5):
        def EEGNet(nb_classes, Chans = 64, Samples = 128, 
            dropoutRate = 0.5, kernLength = 64, F1 = 8, 
            D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
        F2 = D * F1
        EGNet-F_1,D to denote the number of temporal and spatial filters to learn;
        i.e.: EEGNet-4,2 denotes learning 4 temporal filters and 2 spatial filters per temporal filter.
        kernLength: suggest 0.5 * sfreq
        """
        # model = SCCNet(
        #     nb_classes=2, Chans=x_train.shape[1],
        #     Samples=x_train.shape[2], fs=sfreq, dropoutRate=0.5
        # )
        model = EEGNet(
            nb_classes=2, Chans=x_train.shape[1], Samples=x_train.shape[2],
            dropoutRate=0.5, kernLength=math.ceil(0.5 * sfreq),
            F1=8, D=2, F2=16, norm_rate=0.25, dropoutType="Dropout"
        )  # EEGNet-8,2: EEGNet paper suggest in ERN tasks
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=1e-3), metrics=["acc"])
        savedir = os.path.join(os.path.dirname(__file__), "class_md")
        os.makedirs(savedir, exist_ok=True)
        ckpt = MyModelCkpt(savedir, monitor="val_acc", mode="max")
        model.fit(
            x_train, y_train, epochs=400, batch_size=32, shuffle=True,
            validation_data=(x_valid, y_valid), verbose=0, callbacks=[ckpt]
        )

        model = models.load_model(os.path.join(savedir, "sv_weight.h5py"))
        # y_pred = np.argmax(model.predict(x_test), axis=1)
        y_pred = np.argmax(model.predict_on_batch(x_test), axis=1)
        cmat = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        fs = f1_score(y_test, utils.to_categorical(y_pred, 2), average=None)
        auc = roc_auc_score(np.argmax(y_test, axis=1), y_pred)
        csv_writer.writerow([sid, f"T{it}", *fs, auc, *cmat[0, :], *cmat[1, :]])
    csv_writer.writerow([])
fd.close()
    
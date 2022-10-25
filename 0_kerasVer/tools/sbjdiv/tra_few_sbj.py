import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keras_CLEEGN
from keras_CLEEGN.utils.model_tracer import EEG_Viewer
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

x_basepath = "/mnt/left/phlai_DATA/bci-challenge/original"
y_basepath = "/mnt/left/lambert/ner2015_together/fulldata/onlybrain"

def Evaluate(X, y, model, batch_size=32):
    nb_batch = math.ceil(X.shape[0] / batch_size)
    nb_batch = min(nb_batch, 100) # don't use full data at each epoch
    dt_order = np.arange(X.shape[0])
    np.random.shuffle(dt_order)  # shuffling
    X, y = X[dt_order], y[dt_order]

    y_true = y[:batch_size]
    y_pred = model.predict_on_batch(X[:batch_size])
    for bs in range(1, nb_batch):
        indices = (np.array([0, 1], dtype=int) + bs) * batch_size
        y_true = np.append(y_true, y[indices[0]: indices[1]], axis=0)
        y_pred = np.append(y_pred, model.predict_on_batch(X[indices[0]: indices[1]]), axis=0)
    # print(y_true.shape, y_pred.shape)
    return model.loss(y_true, y_pred).numpy()

class Logger:
    log_path = None
    sbuffer_ = ""
    def __init__(self, log_path=None):
        self.log_path = log_path

    def logging(self, msg):
        print("\r{}".format(" " * len(self.sbuffer_)), end="", flush=True)
        print("\r{}".format(msg), end="", flush=True)

        self.sbuffer_ = msg
        if self.log_path is not None:
            fd = open(self.log_path, "a")
            fd.write(msg.strip() + "\n")
            fd.close()

def create_dataset(x_fpath, y_fpath, sbjs, tmin, tmax, win_size=4, stride=2):
    x_raw = mne.io.read_raw_eeglab(x_fpath.format(sbjs[0]), verbose=0)
    sfreq = x_raw.info["sfreq"]
    win_size = math.ceil(win_size * sfreq)
    stride = math.ceil(stride * sfreq)

    tmin = math.ceil(tmin * sfreq)
    tmax = math.ceil(tmax * sfreq)

    X = np.zeros((0, len(x_raw.ch_names), win_size), dtype=np.float32)
    y = np.zeros((0, len(x_raw.ch_names), win_size), dtype=np.float32)
    for sbj in sbjs:
        x_raw = mne.io.read_raw_eeglab(x_fpath.format(sbj), verbose=0)
        y_raw = mne.io.read_raw_eeglab(y_fpath.format(sbj), verbose=0)
        x_content = x_raw[:, tmin: tmax][0]
        y_content = y_raw[:, tmin: tmax][0]

        x_seg = np.array(segment_eeg(x_content, win_size, stride)[0])
        y_seg = np.array(segment_eeg(y_content, win_size, stride)[0])
        X = np.append(X, np.array(x_seg), axis=0)
        y = np.append(y, np.array(y_seg), axis=0)
    return X, y

NUM_SBJS = 12
SAVE_PATH = f"tmpfile/SbjsDiv/{NUM_SBJS}"
NUM_EPOCHS  = 30 #cfg["epochs"]
BATCH_SIZE  = 32
PREFIX = f"{NUM_SBJS}sbj"
os.makedirs(SAVE_PATH, exist_ok=True)

n_trial = 1
ginfo = ["06", "13", "18", "14", "22", "26", "24", "02", "21", "12", "20", "23", "07", "16", "17", "11"]
tra_sbj = ["06", "13", "18", "14", "21", "12", "20", "23", "07", "16", "17", "11"]
tst_sbj = ["22", "26", "24", "02"]
for si in range(0, len(tra_sbj), NUM_SBJS):
    sbj_fold = []
    while len(sbj_fold) != NUM_SBJS:
        sbj_fold.append(tra_sbj[si])
        si += 1
    x_train, y_train = create_dataset(
        os.path.join(x_basepath, "Data_S{}.set"),
        os.path.join(y_basepath, "Data_S{}_Sess.set"),
        sbj_fold, tmin=0, tmax=1800, win_size=4, stride=2
    )
    x_train = np.expand_dims(x_train, axis=3)
    y_train = np.expand_dims(y_train, axis=3)
    x_valid, y_valid = create_dataset(
        os.path.join(x_basepath, "Data_S{}.set"),
        os.path.join(y_basepath, "Data_S{}_Sess.set"),
        sbj_fold, tmin=1800, tmax=2400, win_size=4, stride=2
    )
    x_valid = np.expand_dims(x_valid, axis=3)
    y_valid = np.expand_dims(y_valid, axis=3)
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)

    norm_coef = 1e6
    x_train, y_train = x_train * norm_coef, y_train * norm_coef
    x_valid, y_valid = x_valid * norm_coef, y_valid * norm_coef

    logger = Logger()
    """ initializer """
    mse = losses.MeanSquaredError()
    tra_order    = np.arange(x_train.shape[0])
    nb_batch     = math.ceil(x_train.shape[0] / BATCH_SIZE)
    nb_batch     = min(nb_batch, 100) # don't use full data at each epoch
    min_loss     = mse(x_train, y_train).numpy()
    min_val_loss = mse(x_valid, y_valid).numpy()
    print("min_loss: {:.6f}, min_val_loss: {:.6f}".format(min_loss, min_val_loss))

    n_chan, win_size = x_train.shape[1], x_train.shape[2]
    model = CLEEGN(n_chan=n_chan, duration=win_size, N_F=n_chan, fs=128.0)
    opt = optimizers.Adam(
        learning_rate=optimizers.schedules.ExponentialDecay(
            1e-3, decay_steps=nb_batch, decay_rate=0.98
        )
    )
    model.compile(loss=mse, optimizer=opt, metrics=["acc"])

    loss, val_loss = min_loss, min_val_loss
    loss_curve = {"iter": [], "loss": [], "val_loss": []}
    for ep in range(NUM_EPOCHS):
        np.random.shuffle(tra_order) # shuffle training order
        x_train_ = x_train[tra_order]; y_train_ = y_train[tra_order]
        TimeStpBegin = time.time()
        for bs in range(nb_batch):
            loss, _ = model.train_on_batch(
                x_train_[bs * BATCH_SIZE: (bs + 1) * BATCH_SIZE], 
                y_train_[bs * BATCH_SIZE: (bs + 1) * BATCH_SIZE]
            )
            logger.logging(
                "\rEpoch {}: {}/{} - {:.4f}s - loss={:.6f}".format(
                    ep + 1, bs + 1, nb_batch, time.time() - TimeStpBegin, loss
                )
            )
            
            if (ep * nb_batch + bs) % 20 == 0 or bs == nb_batch - 1:
                """ Evaluate the model each 20 iteration """
                loss = Evaluate(x_train, y_train, model, batch_size=BATCH_SIZE)
                val_loss = Evaluate(x_valid, y_valid, model, batch_size=BATCH_SIZE)
                if loss < min_loss:
                    min_loss = loss
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    model.save(os.path.join(SAVE_PATH, f"{PREFIX}-{n_trial}.h5"))
                loss_curve["iter"].append(ep * nb_batch + bs)
                loss_curve["loss"].append(loss)
                loss_curve["val_loss"].append(val_loss)
            ### End_Of_Batch
        logger.logging(
            "\rEpoch {}/{} - {:.4f}s - min_loss={:.6f} - min_val_loss={:.6f}\n".format(
                ep + 1, NUM_EPOCHS, time.time() - TimeStpBegin, min_loss, min_val_loss
            )
        )
        ### End_Of_Epoch
        print("")
    ### End_Of_Train
    print("min_loss: {:.6f}, min_val_loss: {:.6f}".format(min_loss, min_val_loss))
    scipy.io.savemat(os.path.join(SAVE_PATH, f"loss_curve-{n_trial}.mat"), loss_curve)
    n_trial += 1

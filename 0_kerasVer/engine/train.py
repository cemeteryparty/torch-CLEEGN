import sys
import os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import engine  # noqa: F401
    __package__ = "engine"

import keras_CLEEGN
from keras_CLEEGN.utils.dataset_creater import create_from_annotation
from keras_CLEEGN.models.cleegn import CLEEGN
from keras_CLEEGN.klosses import RootMeanSquaredError

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers

from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import json
import math
import time
import mne

# BCI_CHALL_DIR = "/mnt/left/phlai_DATA/bci-challenge/"
# BCI_CHALL_EVENT = "/mnt/left/phlai_DATA/bci-challenge-event/"
# ICA_Brain_ONLY = "/mnt/left/lambert/ner2015_together/fulldata/onlybrain/"

def Evaluate(X, y, model, batch_size=32):
    y_pred = np.zeros(y.shape, dtype=np.float32)
    for idx in range(0, X.shape[0], batch_size):
        y_pred[idx: idx + batch_size] = model.predict_on_batch(
            X[idx: idx + batch_size]
        )
    return model.loss(y, y_pred).numpy()

def read_json(filepath):
    if filepath is None:
        return None
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content 

def write_json(filepath, content):
    with open(filepath, "w") as fd:
        json.dump(content, fd)


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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CLEEGN")
    parser.add_argument("--train-anno", required=True, type=str, help="path to training dataset annotation")
    parser.add_argument("--valid-anno", type=str, help="path to validation dataset annotation")
    parser.add_argument("--config-path", required=True, type=str, help="path to configuration file")
    parser.add_argument("--load-weight", type=str, help="path to pretrained model path")
    args = parser.parse_args()
    # print(args)

    tra_anno = read_json(args.train_anno)
    val_anno = read_json(args.valid_anno)
    cfg = read_json(args.config_path)

    assert cfg["duration-unit"] in ["sample-point", "sec"]
    SFREQ       = tra_anno["sfreq"]
    N_CHAN      = len(tra_anno["ch_names"])
    NUM_EPOCHS  = cfg["epochs"]
    BATCH_SIZE  = cfg["batch-size"]
    SHUFFLE     = cfg["shuffle"]
    SAVE_PATH   = cfg["save-path"]
    WINDOW_SIZE = cfg["window-size"]
    STRIDE      = cfg["stride"]
    PREFIX      = cfg["model_name"]
    if cfg["duration-unit"] == "sec":
        WINDOW_SIZE = math.ceil(WINDOW_SIZE * SFREQ)
        STRIDE      = math.ceil(STRIDE * SFREQ)
    os.makedirs(SAVE_PATH, exist_ok=True)
    logger = Logger()

    x_train, y_train = create_from_annotation(
        tra_anno, cfg, tmin=tra_anno["tmin"], tmax=tra_anno["tmax"]
    )
    x_train = np.expand_dims(x_train, axis=3)
    y_train = np.expand_dims(y_train, axis=3)
    print(x_train.shape, y_train.shape)
    x_valid, y_valid = create_from_annotation(
        val_anno, cfg, tmin=val_anno["tmin"], tmax=val_anno["tmax"]
    )
    x_valid = np.expand_dims(x_valid, axis=3)
    y_valid = np.expand_dims(y_valid, axis=3)
    print(x_valid.shape, y_valid.shape)

    """ initializer """
    mse = losses.MeanSquaredError()
    tra_order    = np.arange(x_train.shape[0])
    nb_batch     = math.ceil(x_train.shape[0] / BATCH_SIZE)
    min_loss     = mse(x_train, y_train).numpy()
    min_val_loss = mse(x_valid, y_valid).numpy()
    print("min_loss: {:.6f}, min_val_loss: {:.6f}".format(min_loss, min_val_loss))

    """ define model """
    model = CLEEGN(n_chan=N_CHAN, duration=WINDOW_SIZE, N_F=N_CHAN, fs=128.0)
    opt = optimizers.Adam(learning_rate=1e-3)
    # optimizers.schedules.ExponentialDecay(1e-3, decay_steps=nb_batch, decay_rate=0.96)
    model.compile(loss=mse, optimizer=opt, metrics=["acc"])

    loss, val_loss = min_loss, min_val_loss
    for ep in range(NUM_EPOCHS):
        if SHUFFLE:
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
        """ Evaluate """
        loss = Evaluate(x_train, y_train, model, batch_size=BATCH_SIZE)
        val_loss = Evaluate(x_valid, y_valid, model, batch_size=BATCH_SIZE)
        if loss < min_loss:
            min_loss = loss
            model.save(os.path.join(SAVE_PATH, f"{PREFIX}_tra.h5"))
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model.save(os.path.join(SAVE_PATH, f"{PREFIX}_val.h5"))
        logger.logging(
            "\rEpoch {}/{} - {:.4f}s - loss={:.6f} - val_loss={:.6f}\n".format(
                ep + 1, NUM_EPOCHS, time.time() - TimeStpBegin, loss, val_loss
            )
        )
        ### End_Of_Epoch
        print("")
    ### End_Of_Train
    print("min_loss: {:.6f}, min_val_loss: {:.6f}".format(min_loss, min_val_loss))
    # # savemat(basepath + f"{PREFIX}.mat", logs)

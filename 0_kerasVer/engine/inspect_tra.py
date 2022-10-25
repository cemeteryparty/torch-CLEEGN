import sys
import os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import engine  # noqa: F401
    __package__ = "engine"

import keras_CLEEGN
from keras_CLEEGN.utils.dataset_creater import create_from_annotation
from keras_CLEEGN.utils.process_eeg import ar_through_cleegn
from keras_CLEEGN.utils.model_tracer import EEG_Viewer
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
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"{SAVE_PATH} not found")
    img_path = os.path.join(SAVE_PATH, "images")
    os.makedirs(img_path, exist_ok=True)
    logger = Logger()
    viewer = EEG_Viewer(
        os.path.join(tra_anno["x_basepath"], tra_anno["x_fpath"]),
        os.path.join(tra_anno["y_basepath"], tra_anno["y_fpath"]),
        tra_anno["ch_names"]
    )

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
    nb_batch     = min(nb_batch, 100) # don't use full data at each epoch
    min_loss     = mse(x_train, y_train).numpy()
    min_val_loss = mse(x_valid, y_valid).numpy()
    print("min_loss: {:.6f}, min_val_loss: {:.6f}".format(min_loss, min_val_loss))

    """ define model """
    N_CHAN = x_train.shape[1]
    model = CLEEGN(n_chan=N_CHAN, duration=WINDOW_SIZE, N_F=N_CHAN, fs=128.0)
    # opt = optimizers.Adam(learning_rate=1e-3)
    opt = optimizers.Adam(
        learning_rate=optimizers.schedules.ExponentialDecay(
            1e-3, decay_steps=nb_batch, decay_rate=0.98
        )
    )
    model.compile(loss=mse, optimizer=opt, metrics=["acc"])

    loss, val_loss = min_loss, min_val_loss
    loss_curve = {"iter": [], "loss": [], "val_loss": []}
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
            
            if (ep * nb_batch + bs) % 20 == 0 or bs == nb_batch - 1:
                """ Evaluate the model each 20 iteration """
                loss = Evaluate(x_train, y_train, model, batch_size=BATCH_SIZE)
                val_loss = Evaluate(x_valid, y_valid, model, batch_size=BATCH_SIZE)
                if loss < min_loss:
                    min_loss = loss
                    # model.save(os.path.join(SAVE_PATH, f"{PREFIX}_tra.h5"))
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    model.save(os.path.join(SAVE_PATH, f"{PREFIX}_val.h5"))
                loss_curve["iter"].append(ep * nb_batch + bs)
                loss_curve["loss"].append(loss)
                loss_curve["val_loss"].append(val_loss)
            if ep + bs == 0 or bs == nb_batch - 1:
                viewer.view(
                    model, os.path.join(
                        img_path, "{}_iter.png".format(ep * nb_batch + bs)
                    )
                )
        ### End_Of_Batch
        logger.logging(
            "\rEpoch {}/{} - {:.4f}s - min_loss={:.6f} - min_val_loss={:.6f}\n".format(
                ep + 1, NUM_EPOCHS, time.time() - TimeStpBegin, min_loss, min_val_loss
            )
        )
    ### End_Of_Epoch

    print("\nmin_loss: {:.6f}, min_val_loss: {:.6f}".format(min_loss, min_val_loss))
    scipy.io.savemat(os.path.join(SAVE_PATH, "loss_curve.mat"), loss_curve)

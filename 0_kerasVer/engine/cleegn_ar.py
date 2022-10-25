import sys
import os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import engine  # noqa: F401
    __package__ = "engine"

import keras_CLEEGN
from keras_CLEEGN.utils.process_eeg import ar_through_cleegn

from tensorflow.keras import models
from tensorflow.keras import losses

import numpy as np
import json
import math
import time
import mne

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="perform artifact removal on EEG session through CLEEGN")
    parser.add_argument("--eeg-data", required=True, type=str, help="path to nosiy eeg data")
    parser.add_argument("--load-weight", type=str, help="path to pretrained model path")
    parser.add_argument("--tmin", type=float, default=0.0, help="clip eeg data from TMIN (sec)")
    parser.add_argument("--tmax", type=float, default=None, help="clip eeg data end at TMAX (sec)")
    args = parser.parse_args()
    print(args)

    model = models.load_model(args.load_weight)

    ori_raw = mne.io.read_raw_eeglab(args.eeg_data, verbose=0)
    ori_content = ori_raw.get_data() * 1e6

    sfreq = ori_raw.info["sfreq"]
    electrode = ori_raw.ch_names
    tmin = math.floor(args.tmin * sfreq)
    tmax = ori_content.shape[1] if args.tmax is None else math.ceil(args.tmax * sfreq)
    stride = math.ceil(0.5 * sfreq)

    model_input_shape = model.layers[0].input_shape[0][1:-1]
    if len(electrode) != model_input_shape[0]:
        if model_input_shape[0] == 8:
            ch_names = ["Fp1", "Fp2", "T7", "T8", "O1", "O2", "Fz", "Pz"]
            select_idx = [electrode.index(ch_name) for ch_name in ch_names]
        elif model_input_shape[0] == 19:
            pass
        ori_content = ori_content[select_idx]

    x_content = ori_content[:, tmin: tmax]

    t0 = time.time()
    y = ar_through_cleegn(x_content, model, stride)
    print(time.time() - t0)

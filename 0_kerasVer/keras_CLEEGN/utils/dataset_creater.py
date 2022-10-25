""" Dataset Generate Function """
from .process_eeg import segment_eeg
import numpy as np
import math
import mne
import os

def create_from_annotation(anno_obj, cfg, tmin=0, tmax=None):
    """ create training dataset from annotation
    
    Args
        anno_obj : A annotation object (dict)
        cfg      : A configuration object (dict)
        tmin     : The first time sample (sec) to include.
        tmax     : End time sample (sec). If None (default), the end of the data is used.
    Returns
        X, y : Pairing of x (attribute) and y (ground-truth)
    """
    x_fpath = os.path.join(anno_obj["x_basepath"], anno_obj["x_fpath"])
    y_fpath = os.path.join(anno_obj["y_basepath"], anno_obj["y_fpath"])

    WINDOW_SIZE = math.ceil(cfg["window-size"])
    STRIDE      = math.ceil(cfg["stride"])
    SFREQ       = anno_obj["sfreq"]
    SELECTS     = anno_obj["ch_names"]
    if cfg["duration-unit"] == "sec":
        WINDOW_SIZE = math.ceil(WINDOW_SIZE * SFREQ)
        STRIDE      = math.ceil(STRIDE * SFREQ)

    sfreq = anno_obj["sfreq"]
    X, y = [], []
    for fmt_term in anno_obj["fmt-terms"]:
        # read mne object
        x_raw = mne.io.read_raw_eeglab(x_fpath.format(*fmt_term))
        y_raw = mne.io.read_raw_eeglab(y_fpath.format(*fmt_term))

        norm_coef = 1e-6  # x10^6
        x_content = x_raw.get_data() / norm_coef
        y_content = y_raw.get_data() / norm_coef

        if SELECTS is not None:
            select_idx = [x_raw.ch_names.index(ch_name) for ch_name in SELECTS]
            x_content = x_content[select_idx]
            y_content = y_content[select_idx]

        if tmax is None:
            x_content = x_content[:, math.ceil(tmin * sfreq):]
            y_content = y_content[:, math.ceil(tmin * sfreq):]
        else:
            x_content = x_content[:, math.ceil(tmin * sfreq): math.ceil(tmax * sfreq)]
            y_content = y_content[:, math.ceil(tmin * sfreq): math.ceil(tmax * sfreq)]

        x_seg, _ = segment_eeg(x_content, WINDOW_SIZE, STRIDE)
        y_seg, _ = segment_eeg(y_content, WINDOW_SIZE, STRIDE)
        X += x_seg
        y += y_seg

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

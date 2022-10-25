""" Analyze Reconstruction and Artifact Removal Ability """
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keras_CLEEGN
from keras_CLEEGN.utils.dataset_creater import create_from_annotation
from keras_CLEEGN.utils.process_eeg import ar_through_cleegn
from keras_CLEEGN.klosses import RootMeanSquaredError
from keras_CLEEGN.models.cleegn import CLEEGN

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import losses

from scipy.io import loadmat
import numpy as np
import json
import math
import time
import mne

import plotly.graph_objects as go
from plotly import tools
import plotly as ply

BCI_CHALL_EVENT = "/mnt/left/phlai_DATA/bci-challenge-event/"

def viewARA(tstmps, data_colle, ref_i, electrode, titles=None, colors=None, alphas=None):
    n_data = len(data_colle)
    titles = ["" for di in range(n_data)] if titles is None else titles
    alphas = [0.5 for di in range(n_data)] if alphas is None else alphas
    if colors is None:
        cmap_ = plt.cm.get_cmap("tab20", n_data)
        colors = [rgb2hex(cmap_(di)) for di in range(n_data)]

    picks_chs = ["Fp1", "Fp2", "T7", "T8", "O1", "O2", "Fz", "Pz"]
    picks = [electrode.index(c) for c in picks_chs]
    for di in range(n_data):
        data_colle[di] = data_colle[di][picks, :]

    kwargs = {"showgrid": False, "showticklabels": False, "zeroline": False}
    layout = go.Layout(font={"size": 15}, autosize=False, width=1000, height=600)
    layout.update({"xaxis": go.layout.XAxis(kwargs)})
    kwargs.update(range=[-0.5, len(picks) - 0.5])
    layout.update({"yaxis": go.layout.YAxis(kwargs)})
    annos, traces = [], []
    for ii, ch_name in enumerate(picks_chs):
        offset = len(picks) - ii - 1
        norm_coef = 0.25 / np.abs(data_colle[ref_i][ii]).max()

        for di in range(n_data):
            eeg_dt = data_colle[di]
            traces.append(go.Scatter(
                x=tstmps, y=eeg_dt[ii] * norm_coef + offset,
                name=titles[di], line_color=colors[di], opacity=alphas[di],
                showlegend=(False if ii else True)
            ))
        annos.append(go.layout.Annotation(
            x=-(50 / layout["width"]), y=offset, text=ch_name, xref="paper", showarrow=False
        ))  # add ch_names using Annotations
    layout.update(annotations=annos)
    layout.update(legend={
        "orientation": "h", "xanchor": "right", "yanchor": "bottom", "y": 1.02, "x": 1
    })
    return go.Figure(data=traces, layout=layout)

def read_json(filepath):
    if filepath is None:
        return None
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content 

sid = "02" # 02, 14, 20
ginfo = ["06", "13", "18", "14"] + ["22", "26", "24", "02"] + ["21", "12", "20", "23"] + ["07", "16", "17", "11"]
basepath = "/mnt/left/home/2022/phlai410277/keras-CLEEGN/tmpfile/sv"
trial_name = "bc-12_0001.12_3040.4"
basepath = os.path.join(basepath, trial_name)
savedir =  os.path.join(os.path.expanduser("~"), "Downloads")

anno_path = os.path.join(basepath, "set_{}/set_test.json".format(gid))
conf_path = os.path.join(basepath, "set_{}/config.json".format(gid))
anno = read_json(anno_path)
cfg = read_json(conf_path)
assert cfg["duration-unit"] in ["sample-point", "sec"]
picks     = anno["ch_names"]
model_path = os.path.join(
    basepath, "set_{}/{}_val.h5".format(gid, cfg["model_name"])
)

x_fpath = os.path.join(anno["x_basepath"], anno["x_fpath"])
y_fpath = os.path.join(anno["y_basepath"], anno["y_fpath"])
x_raw = mne.io.read_raw_eeglab(x_fpath.format(sid), verbose=0)
y_raw = mne.io.read_raw_eeglab(y_fpath.format(sid), verbose=0)
mne.rename_channels(x_raw.info, mapping={"P08": "PO8"})
mne.rename_channels(y_raw.info, mapping={"P08": "PO8"})
sfreq = x_raw.info["sfreq"]
electrode = x_raw.ch_names

events = loadmat(os.path.join(BCI_CHALL_EVENT, f"Data_S{sid}_event.mat"))["events"]
centr = events[0][0]
tmin, tmax = math.ceil(0 * sfreq), math.ceil(10 * sfreq)

x_content, tstmps = x_raw[:, centr - tmin: centr + tmax]
y_content, tstmps = y_raw[:, centr - tmin: centr + tmax]
x_content *= 1e6
y_content *= 1e6
gid = ginfo.index(sid) // 4 + 1
model_path = os.path.join(
    basepath, "set_{}/{}_val.h5".format(gid, cfg["model_name"])
)
model = models.load_model(model_path)
rdelta = max(tmin, tmax) * 2
y = ar_through_cleegn(
    x_raw[:, centr - rdelta: centr + rdelta][0] * 1e6,
    model, math.ceil(1.0 * sfreq)
)
y = y[:, rdelta - tmin: rdelta + tmax]
print(x_content.shape, y_content.shape, y.shape)

fig = viewARA(
    tstmps, [x_content, y_content, y], 1, electrode,
    titles=["", "", ""], colors=["black", "black", "green"], alphas=[0, 0.8, 0]
)
fig.write_image(os.path.join(
    savedir, f"S{sid}_{trial_name}.png"
))

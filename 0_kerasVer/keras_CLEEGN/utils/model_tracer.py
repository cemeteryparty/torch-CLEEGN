from .process_eeg import ar_through_cleegn
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from plotly import tools
# import plotly as ply
import numpy as np
import math
import mne
import os


class EEG_Viewer:
    def __init__(self, x_fpath, y_fpath, picks=None):
        x_raw = mne.io.read_raw_eeglab(x_fpath.format("02"), verbose=0)
        y_raw = mne.io.read_raw_eeglab(y_fpath.format("02"), verbose=0)
        mne.rename_channels(x_raw.info, mapping={"P08": "PO8"})
        mne.rename_channels(y_raw.info, mapping={"P08": "PO8"})
        self.SFREQ = x_raw.info["sfreq"]
        self.STRIDE = math.ceil(1 * self.SFREQ)
        self.electrode = x_raw.ch_names
        select_idx = range(len(x_raw.ch_names))
        if picks is not None:
            select_idx = [x_raw.ch_names.index(ch_name) for ch_name in picks]
            self.electrode = picks

        sfreq = x_raw.info["sfreq"]
        tmin, tmax = math.ceil(8.0 * sfreq), math.ceil(13.0 * sfreq)
        self.x_content, self.tstmps = x_raw[select_idx, tmin: tmax]
        self.x_content *= 1e6
        self.y_content = y_raw[select_idx, tmin: tmax][0] * 1e6

    def view(self, model, img_name):
        picks_chs = ["Fp1", "T7", "Cz", "T8", "O2"]
        picks = [self.electrode.index(c) for c in picks_chs]
        y = ar_through_cleegn(self.x_content, model, self.STRIDE)
        data_colle = [self.x_content[picks, :], self.y_content[picks, :], y[picks, :]]
        n_data, ref_i = len(data_colle), 1
        colors, alphas = ["#929591", "#ff0000", "#0000ff"], [0.3, 0.5, 0.5]
        titles = ["Raw", "Offline_Method", "CLEEGN"]
        width = [1.5, 2, 2.5]

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        for ii, ch_name in enumerate(picks_chs):
            offset = len(picks) - ii - 1
            norm_coef = 0.25 / np.abs(data_colle[ref_i][ii]).max()
            for di in range(n_data):
                eeg_dt = data_colle[di]
                ax.plot(
                    self.tstmps, eeg_dt[ii] * norm_coef + offset, label=None if ii else titles[di],
                    color=colors[di], alpha=alphas[di], linewidth=width[di]
                )
        ax.set_xlim(self.tstmps[0], self.tstmps[-1])
        ax.set_ylim(-0.5, len(picks) - 0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        # ax.tick_params(axis="x", labelsize=20)
        # ax.set_yticks(np.arange(len(picks)))
        # ax.set_yticklabels(picks_chs[::-1], fontsize=25)

        # ax.legend(
        #     bbox_to_anchor=(0, 1.02, 1, 0.2),
        #     loc="lower right", borderaxespad=0, ncol=3, fontsize=20
        # )
        plt.savefig(img_name)
        plt.close("all")
    """
    def view(self, model, img_name):
        picks_chs = ["Fp1", "Fp2", "T7", "T8", "O1", "O2", "Fz", "Pz"]
        picks = [self.electrode.index(c) for c in picks_chs]
        tstmps = self.tstmps
        y = ar_through_cleegn(self.x_content, model, self.STRIDE)
        data_colle = [self.x_content[picks], self.y_content[picks], y[picks]]
        titles = ["Raw", "Offline_Method", "CLEEGN"]
        colors = ["gray", "red", "blue"]
        alphas = [0.3, 0.5, 0.5]
        n_data = len(data_colle)

        kwargs = {"showgrid": False, "showticklabels": False, "zeroline": False}
        layout = go.Layout(font={"size": 15}, autosize=False, width=1000, height=600)
        layout.update({"xaxis": go.layout.XAxis(kwargs)})
        kwargs.update(range=[-0.5, len(picks) - 0.5])
        layout.update({"yaxis": go.layout.YAxis(kwargs)})
        annos, traces = [], []
        for ii, ch_name in enumerate(picks_chs):
            offset = len(picks) - ii - 1
            norm_coef = 0.25 / np.abs(data_colle[1][ii]).max()

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
        layout.update(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ))
        fig = go.Figure(data=traces, layout=layout)
        fig.write_image(img_name)
    """

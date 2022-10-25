""" Process EEG signal """
from tensorflow.keras import models
from scipy import signal
import numpy as np
import mne

def segment_eeg(eeg, window_size=100, stride=50):
    """ Session EEG Signal by Slinding Window """
    n_chan, n_timep = eeg.shape
    tstamps = []
    segments = []
    for i in range(0, n_timep, stride):
        seg = eeg[:,i: i + window_size]
        if seg.shape != (n_chan, window_size):
            break
        segments.append(seg)
        tstamps.append(i)

    return segments, tstamps

def ar_through_cleegn(eeg_data, model, stride):
    """ perform artifact removal on EEG session through CLEEGN
    
    Args
        eeg_data : EEG data in shape (n_chan, sample_points) (numpy.ndarray)
        model    : CLEEGN model (keras.models.Model)
        stride   : Stride of the sliding window (nb_sample_point)
    Returns
        noiseless_eeg : reconstructed noiseless EEG data
    """
    model_input_shape = model.layers[0].input_shape[0][1:3] # tuple
    if model_input_shape[0] != eeg_data.shape[0]:
        raise ValueError("shape unmatched between EEG data and model inputs.")
    window_size = model_input_shape[1]


    noiseless_eeg = np.zeros(eeg_data.shape, dtype=np.float32)
    hcoef = np.zeros(eeg_data.shape[1], dtype=np.float32)

    hwin = signal.hann(window_size) + 1e-9
    for i in range(0, noiseless_eeg.shape[1], stride):
        tstap, LAST_FRAME = i, False
        segment = eeg_data[:, tstap: tstap + window_size]
        if segment.shape[1] != window_size:
            tstap = noiseless_eeg.shape[1] - window_size
            segment = eeg_data[:, tstap:]
            LAST_FRAME = True

        pred_segment = model.predict_on_batch(
            np.expand_dims(np.expand_dims(segment, axis=0), axis=-1)
        )
        noiseless_eeg[:, tstap: tstap + window_size] += \
            pred_segment.squeeze() * hwin
        hcoef[tstap: tstap + window_size] += hwin

        if LAST_FRAME:
            break
    noiseless_eeg /= hcoef

    return noiseless_eeg

"""
import multiprocessing as mp
class MP_Method(mp.Process):
    def  __init__(self, iop, nline, task, in_src):
        mp.Process.__init__(self)
        self.daemon = True
        self.iop = iop # id of process
        self.nline = nline
        self.task = task
        self.in_src = in_src
    def run(self):
        i = self.iop
        while i < len(self.in_src):
            try:
                if self.task == 1:
                    ProofofWork(self.in_src[i])
                i += self.nline
            except KeyboardInterrupt:
                return
def main():
    line = input().split(" ")
    method = int(line[0])
    if method == 1 or method == 2:
        nline = int(line[1])
    n = int(input())
    for _ in range(n):
        inputs.append(input())

    proc_start = Timer()
    pList = []
    for i in range(nline):
        subtask = MP_Method(i, nline, task, inputs)
        subtask.start()
        pList.append(subtask)
    while (True in [p.is_alive() for p in pList]):
        try:
            [p.join(1.0) for p in pList if p and p.is_alive()]
        except KeyboardInterrupt:
            [p.terminate() for p in pList]
            exit(1)

    proc_end = Timer()
    print(f"\nProccess time: {proc_end - proc_start} sec")
"""

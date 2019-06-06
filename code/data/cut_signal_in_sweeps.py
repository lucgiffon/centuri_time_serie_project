import pyabf
import pathlib
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy import fft, ifft
import pandas as pd
import numpy as np
import copy
from collections import OrderedDict

from centuri_project.utils import interim_directory, save_data_to_folder, processed_directory

if __name__ == "__main__":
    filenames = ["train.npz", "test.npz"]
    nb_sweep_by_trace = 9
    for filename in filenames:
        data_file = interim_directory / filename
        data = np.load(data_file, allow_pickle=True)
        lst_idx_sweeps = np.split(np.arange(data["signals"].shape[1]), nb_sweep_by_trace)
        save_data_to_folder(
            np.vstack([data["signal_times"][:, idxs] for idxs in lst_idx_sweeps]),
            np.repeat(data["sampling_rates"], nb_sweep_by_trace),
            np.vstack([data["signals"][:, idxs] for idxs in lst_idx_sweeps]),
            np.vstack([data["labels"][:, :, idxs] for idxs in lst_idx_sweeps]) if filename == "train.npz" else None,
            filename,
            processed_directory
        )


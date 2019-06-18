import pyabf
import pathlib
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy import fft, ifft
import pandas as pd
import numpy as np
import copy
from collections import OrderedDict

from centuri_project.utils import save_data_to_folder, interim_directory, raw_data_directory


def running_mean(x, N):
    """
    Compute running mean of
    :param x:
    :param N:
    :return:
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def butter_lowpass(cutoff, sampling_frequency, order=5):
    """
    Create butter lowpass filter

    :param cutoff: frequencies greater than cutoff will be cutted off.
    :param sampling_frequency: the number of sample each second
    :param order: order of the filter
    :return:
    """
    nyq = 0.5 * sampling_frequency  # nyquist frequency
    normal_cutoff = cutoff / nyq  #
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output="ba")
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply butter lowpass filter to data

    :param data: The data (amplitude: Y axis) to filter
    :param cutoff:
    :param fs:
    :param order:
    :return:
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def band_stop_fft_filter(data, freq_range_stop, ):
    fft_data = fft(data)
    bandstoped_fft_data = fft_data[:]
    for i in range(*freq_range_stop):
        bandstoped_fft_data[i] = 0
    # plt.plot(fft_data)
    # plt.show()
    filtered_data = ifft(bandstoped_fft_data)
    return filtered_data

def process_abf_data(abf_data, ax=None):
    """
    Apply filtering on abf data:
        * butterworth filtering attenuate > 1000 Hz frequencies;
        * fft band-stop filtering remove 50 Hz frequency.

    :param abf_data: Signal amplitude values
    :return: The filtered abfdata
    """
    cutoff_freq_1000 = 1000  # Hz, the frequencies gt to be cuted of
    order = 6 # ???

    sampling_rate = abf_data.dataRate  # Hz

    # filter out high frequencies > 1000 Hz
    filtered_abf_data_Y = butter_lowpass_filter(abf_data.sweepY, cutoff_freq_1000, sampling_rate, order)

    # filter out ac frequencies (between 50 and 60 Hz)
    ac_filtered_abf_data_Y = band_stop_fft_filter(filtered_abf_data_Y, (50, 51))

    # remove frames of values around big spike
    # indexes_values_outside_limits = ac_filtered_abf_data_Y[ac_filtered_abf_data_Y < miny or ac_filtered_abf_data_Y > maxy]

    if ax is not None:
        ax.plot(abf_data.sweepX, abf_data.sweepY, color="grey", zorder=-2, label="raw data")
        ax.plot(abf_data.sweepX, filtered_abf_data_Y, color="b", zorder=-1, label="high frequencies filtered data")
        ax.plot(abf_data.sweepX, ac_filtered_abf_data_Y, color="r", zorder=0, label="AC filtered data")
        # ax.plot(abf_data.sweepX, ac_filtered_abf_data_Y, color="grey", zorder=0, label="Signal data")

    return ac_filtered_abf_data_Y

def process_asc_file(file, signal_size, signal_sampling_rate, ax=None):
    """
    Read asc file in pandas dataframe and extract fields of interest:
        * idx 1: time of events -> converted to one hot over all sampling times
        * idx 2: amplitudes of events
        * idx 6: baselines of events

    :param file:
    :return:
    """
    asc_data = pd.read_csv(file, delimiter="\t", header=None)
    asc_data[1] = asc_data[1].apply(lambda str_coma_float: float(str_coma_float.replace(",", "")) * 1e-3)
    # get interesting fields in pandas as numpy arrays (all the same dimension)
    all_events_time = asc_data[1].values
    all_amplitudes_at_events = asc_data[2].values
    all_baselines_at_events = asc_data[6].values

    # create sparse vectors for each interesting field
    events_indexes = (all_events_time * signal_sampling_rate).astype(np.int)
    one_hot_event = np.zeros(signal_size)
    one_hot_event[events_indexes] = 1
    amplitudes_events = np.zeros(signal_size)
    amplitudes_events[events_indexes] = all_amplitudes_at_events
    baseline_events = np.zeros(signal_size)
    baseline_events[events_indexes] = all_baselines_at_events
    # concatenate those sparse vactors into a cube
    cube_events = np.stack([one_hot_event, amplitudes_events, baseline_events], axis=0)


    if ax is not None:
        ax.scatter(all_events_time, all_baselines_at_events - all_amplitudes_at_events, edgecolors="r", facecolors="none", s=80, zorder=1, label="event")
        ax.scatter(all_events_time, all_baselines_at_events - all_amplitudes_at_events, edgecolors="r", facecolors="none", s=80, zorder=1, label="event")
        ax.scatter(all_events_time, all_baselines_at_events, color='y', s=80, marker='x', zorder=2, label="baseline signal")
        # ax.scatter(all_events_time, all_baselines_at_events - all_amplitudes_at_events, color="b", marker="x", s=80, zorder=1, label="event")

    return cube_events

def get_filtered_signals():
    abf_train_files = [file for file in raw_data_directory.glob("**/*") if file.is_file()]

    lst_test = {}
    lst_test["signals"] = []
    lst_test["signals_times"] = []
    lst_test["sampling_rates"] = []
    lst_test["names"] = []
    lst_train = {}
    lst_train["signals"] = []
    lst_train["signals_times"] = []
    lst_train["cube_labels"] = []
    lst_train["sampling_rates"] = []
    lst_train["names"] = []
    for file in abf_train_files:
        if file.suffix.upper() == ".ABF":
            abf_file = file
            abf_data = pyabf.ABF(abf_file)
            sampling_rate = abf_data.dataRate
            if sampling_rate != 10000: # when sweeps have sampling rates different than 10000, the sweep has not the same length, ignoring them atm
                continue
            # filter data (ac noise frequency (50 Hz) and > 1000 Hz frequencies)
            filtered_abf_data_Y = process_abf_data(abf_data)

            lst_test["signals"].append(filtered_abf_data_Y)
            lst_test["signals_times"].append(abf_data.sweepX)
            lst_test["sampling_rates"].append(sampling_rate)
            lst_test["names"].append(abf_file.stem)

        elif file.suffix.upper() == ".ASC":
            f, ax = plt.subplots()

            # get data in abf file
            abf_file = file.with_suffix('.ABF')
            abf_data = pyabf.ABF(abf_file)
            sampling_rate = abf_data.dataRate

            # filter data (ac noise frequency (50 Hz) and > 1000 Hz frequencies)
            filtered_abf_data_Y = process_abf_data(abf_data, ax)

            # get the cube of dim n x d x 3 of labels. The 3 labels are: event presence with 1hot vector, amplitude of event, baseline of event
            asc_file = file
            # cube_labels = process_asc_file(asc_file, len(filtered_abf_data_Y), sampling_rate, ax)
            cube_labels = process_asc_file(asc_file, len(filtered_abf_data_Y), sampling_rate, None)


            lst_train["signals"].append(filtered_abf_data_Y)
            lst_train["signals_times"].append(abf_data.sweepX)
            lst_train["cube_labels"].append(cube_labels)
            lst_train["sampling_rates"].append(sampling_rate)
            lst_train["names"].append(abf_file.stem)


            # vizualization settings
            time_win_mean = 0.5
            offset_y = 50
            running_mean_win = int(time_win_mean * sampling_rate)
            runing_mean_abf_dataY = running_mean(filtered_abf_data_Y, running_mean_win)
            maxy = max(runing_mean_abf_dataY) + offset_y
            miny = min(runing_mean_abf_dataY) - offset_y
            limits = (miny, maxy)
            ax.set_ylim(*limits)
            f.suptitle(file.name)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))

            plt.legend(by_label.values(), by_label.keys())
            # plt.show()

        else:
            raise ValueError("Unknown file")

    train_data = {
        "signal_times": np.stack(lst_train["signals_times"], axis=0),
        "sampling_rates": np.array(lst_train["sampling_rates"]),
        "signals": np.stack(lst_train["signals"], axis=0),
        "labels": np.stack(lst_train["cube_labels"], axis=0),
        "names": np.array(lst_train["names"])
    }

    test_data = {
        "signal_times": np.stack(lst_test["signals_times"], axis=0),
        "sampling_rates": np.array(lst_train["sampling_rates"]),
        "signals": np.stack(lst_test["signals"], axis=0),
        "labels": None,
        "names": np.array(lst_test["names"])
    }

    return train_data, test_data


if __name__ == "__main__":
    get_filtered_signals()
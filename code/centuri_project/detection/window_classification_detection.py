import itertools
import math
import keras
from keras import Sequential
from keras.layers import Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, Flatten
from keras.utils import to_categorical
from scipy.signal import detrend
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from centuri_project.utils import processed_directory, get_dataset_of_windows, create_window_indexes, trained_models_directory, results_directory
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd


def rle(inarray):
        """
        https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi

        run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            p_pairs = [(p[i], p[i+1]) for i, _ in enumerate(p[:-1])]
            p_pairs.append((p[-1], len(ia)))
            return(z, p_pairs, ia[i])

def compare_label_annotations(labels_1h_predicted, labels_1h_truth, p_window_size):
    extension_label = (p_window_size-1) / 2  # type: float
    extension_label_before = math.floor(extension_label)
    extension_label_after = math.ceil(extension_label)


    for sweep_labels in labels_1h_predicted:

        a, b, c = rle(sweep_labels)
        nbr_of_consecutive_values = a
        indexes_of_consecutive_values = b
        consecutive_values = c

        for idx_val, val in consecutive_values:
            if val != 1: continue
            start, stop = indexes_of_consecutive_values[idx_val]
            extra_start, extra_stop = np.clip(start - extension_label_before, a_min=0, a_max=len(sweep)), np.clip(stop + extension_label_after, a_min=0, a_max=len(sweep))
            np.put(labels_1h_predicted, np.arange(extra_start, extra_stop), 1)
            a = 1

def main():
    data_file = processed_directory / "splitted_train_with_annotations.npz"  # all train data with annotations
    data = np.load(data_file, allow_pickle=True)
    data_signal_values = data["signals"]  # for all trace the value records (n x d matrix): the actual traces we need to deal with
    data_labels = data["labels"]  # for all trace, the event presence (as 1/0 vector), the event amplitude and the baseline (n x 3 x d cube of data)
    data_names = data["names"]

    mapping = {}
    for i in np.unique(data_names):
        mapping[i] = np.where(data_names == i)[0]

    test_names = ['2019_02_27_02-sEPSC', '2017_08_04_00-sEPSC']

    # remove a portion of labeled samples so that we can later evaluate event detection
    test_indices = [item for sublist in [mapping[tst_na] for tst_na in test_names] for item in sublist]
    data_signal_values_test = data_signal_values[test_indices].astype(np.float)[:, :10000]
    data_labels_test = data_labels[test_indices][:, :, :10000]

    print("{} sweeps à annoter".format(len(data_signal_values_test)))

    identifier_event_classifier = "1560219172"
    dir_models = trained_models_directory / "classification/cnn"
    models_ext = ".h5"
    dir_results = results_directory / "classification/cnn"
    results_ext = ".csv"

    df_results = pd.read_csv(dir_results / (identifier_event_classifier + results_ext))
    model = keras.models.load_model((dir_models / ("cnn_" + identifier_event_classifier + models_ext)).absolute().as_posix())

    window_size = 300  # The size of each window (signal cut)
    attention_span = 100  # The attention span in which to look for events in each window
    # |   <--attention span-->   |
    # |<-------window size------>|
    step_between_windows = 1  # the step size between each first index of windows

    threshold_nbr_consecutive_events = 1
    confidence_bound = 0.5

    window_indexes = create_window_indexes(data_signal_values_test.shape[1], window_size, step_between_windows)

    data_labels_test_predicted = np.zeros_like(data_labels_test)
    for idx_sweep, sweep in enumerate(data_signal_values_test):
        sweep_labels = data_labels_test[idx_sweep]
        sweep_windows = sweep[window_indexes]

        sweep_windows = normalize(sweep_windows, axis=1)
        sweep_windows = detrend(sweep_windows, axis=1)
        sweep_windows = np.expand_dims(sweep_windows, axis=-1)

        predictions = model.predict(sweep_windows)
        predictions[:, 1] = np.where(predictions[:, 1] < confidence_bound, np.zeros_like(predictions[:, 1]), predictions[:, 1])

        window_classes = np.argmax(predictions, axis=1)  # best classifier id: 1560219172 . win size = 300. attention_span_size = 100
        nb_zero = (window_size-1) / 2  # type: float
        nb_zero_before = math.floor(nb_zero)
        nb_zero_after = math.ceil(nb_zero)
        window_classes_with_zeros_padding = np.vstack([
            np.zeros((nb_zero_before, 1)),
            window_classes[:, None],
            np.zeros((nb_zero_after, 1))
        ])


        outlabels = np.zeros_like(sweep_labels)  # 3 x d: event-1hot, amplitude, baseline
        a, b, c = rle(window_classes)
        nbr_of_consecutive_values = a
        indexes_of_consecutive_values = b
        consecutive_values = c
        for idx_val, val in enumerate(consecutive_values):
            if val != 1:
                continue

            if nbr_of_consecutive_values[idx_val] < threshold_nbr_consecutive_events:
                continue

            start, stop = indexes_of_consecutive_values[idx_val]
            # extra_start, extra_stop = np.clip(start - window_size, a_min=0, a_max=len(sweep)), np.clip(stop + window_size, a_min=0, a_max=len(sweep))
            extra_start, extra_stop = np.clip(start - attention_span, a_min=0, a_max=len(sweep)), np.clip(stop + attention_span, a_min=0, a_max=len(sweep))
            idx_max_value_in_episode = np.argmin(detrend(sweep[extra_start:extra_stop]))
            absolute_idx_max_value = idx_max_value_in_episode + extra_start
            max_value_in_episode = sweep[absolute_idx_max_value]
            outlabels[0, absolute_idx_max_value] = 1
            outlabels[1, absolute_idx_max_value] = max_value_in_episode
        #
        # plt.plot(sweep, color="lime", label="Input Excitatory signal")
        # plt.scatter(np.arange(len(sweep))[outlabels[0].astype(np.bool)], sweep[outlabels[0].astype(np.bool)], color="r", label="Detected events", zorder=5)
        # plt.scatter(np.arange(len(sweep))[sweep_labels[0].astype(np.bool)], sweep[sweep_labels[0].astype(np.bool)], color="b", marker="x", label="Manually labeled events", zorder=10)
        # plt.show()

        data_labels_test_predicted[idx_sweep] = outlabels

    # compare_label_annotations(data_labels_test_predicted[:, 0, :])

    print("enlever le cap à 10000")

if __name__ == "__main__":
    main()
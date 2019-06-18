import math
import keras
from scipy.signal import detrend

from centuri_project.utils import create_window_indexes, trained_models_directory, results_directory, prepare_test_data, rle
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt


def main():

    data_signal_values_test, data_labels_test = prepare_test_data()
    # note that I only work on the first 10000 data values (for quickness while debugging)
    data_signal_values_test = data_signal_values_test[:, :10000]
    data_labels_test = data_labels_test[:, :, :10000]

    print("{} sweeps à annoter".format(len(data_signal_values_test)))

    # hardcoded parameters for the keras model
    identifier_event_classifier = "1560219172"
    dir_models = trained_models_directory / "classification/cnn"
    models_ext = ".h5"
    dir_results = results_directory / "classification/cnn"
    results_ext = ".csv"

    df_results = pd.read_csv(dir_results / (identifier_event_classifier + results_ext))
    # load Keras model for detection (filename)
    model = keras.models.load_model((dir_models / ("cnn_" + identifier_event_classifier + models_ext)).absolute().as_posix())

    # hardcoded data parameters (should be found in df_results)
    # todo make it dynamic wrt df_results
    window_size = 300  # The size of each window (signal cut)
    attention_span = 100  # The attention span in which to look for events in each window
    # |   <--attention span-->   |
    # |<-------window size------>|
    step_between_windows = 1  # the step size between each first index of windows

    # hardcoded parameters for the sensitivity of the detector (should be crossvalidated)
    threshold_nbr_consecutive_events = 1
    confidence_bound = 0.5

    window_indexes = create_window_indexes(data_signal_values_test.shape[1], window_size, step_between_windows)

    data_labels_test_predicted = np.zeros_like(data_labels_test)
    for idx_sweep, sweep in enumerate(data_signal_values_test):
        sweep_labels = data_labels_test[idx_sweep]
        # get all sucessing windows in the trace (ordered)
        sweep_windows = sweep[window_indexes]

        # preprocess the windows
        sweep_windows = normalize(sweep_windows, axis=1)
        sweep_windows = detrend(sweep_windows, axis=1)
        sweep_windows = np.expand_dims(sweep_windows, axis=-1)

        # get prediction for each window
        predictions = model.predict(sweep_windows)
        predictions[:, 1] = np.where(predictions[:, 1] < confidence_bound, np.zeros_like(predictions[:, 1]), predictions[:, 1])
        window_classes = np.argmax(predictions, axis=1)  # best classifier id: 1560219172 . win size = 300. attention_span_size = 100

        # pad the result with zero so that it has the same size than the input trace (offset introduced by the window size)
        nb_zero = (window_size-1) / 2  # type: float
        nb_zero_before = math.floor(nb_zero)
        nb_zero_after = math.ceil(nb_zero)
        window_classes_with_zeros_padding = np.vstack([
            np.zeros((nb_zero_before, 1)),
            window_classes[:, None],
            np.zeros((nb_zero_after, 1))
        ])


        # build the actual vector of anomaly positions (replace bands of anomaly detection by 1 single most relevant position)
        outlabels = np.zeros_like(sweep_labels)  # 3 x d: event-1hot, amplitude, baseline
        # get the positions and etc. of the bands of 1 and 0 in the input
        a, b, c = rle(window_classes_with_zeros_padding)
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
            # the most relevant position for an anomaly is at the maximum value position in the band corresponding to that anomaly
            idx_max_value_in_episode = np.argmin(detrend(sweep[extra_start:extra_stop]))
            absolute_idx_max_value = idx_max_value_in_episode + extra_start
            max_value_in_episode = sweep[absolute_idx_max_value]
            outlabels[0, absolute_idx_max_value] = 1
            outlabels[1, absolute_idx_max_value] = max_value_in_episode
        #
        plt.plot(sweep, color="grey", label="Input Excitatory signal")
        plt.scatter(np.arange(len(sweep))[outlabels[0].astype(np.bool)], sweep[outlabels[0].astype(np.bool)],edgecolors="red", facecolors="none", color="r", s=80, label="Detected events", zorder=5)
        plt.scatter(np.arange(len(sweep))[sweep_labels[0].astype(np.bool)], sweep[sweep_labels[0].astype(np.bool)], color="b", s=80, marker="x", label="Manually labeled events", zorder=10)
        plt.legend()
        plt.show()

        data_labels_test_predicted[idx_sweep] = outlabels

    # compare_label_annotations(data_labels_test_predicted[:, 0, :])

    print("enlever le cap à 10000")

if __name__ == "__main__":
    main()
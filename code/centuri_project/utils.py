import math

import pathlib
import numpy as np


project_dir = pathlib.Path(__file__).resolve().parents[2]

raw_data_directory = project_dir / pathlib.Path("data/raw/1_time_series_analysis/EPSC/Traces & Events")
interim_directory = project_dir / pathlib.Path("data/interim")
processed_directory = project_dir / pathlib.Path("data/processed")

trained_models_directory = project_dir / "models/trained"

results_directory = project_dir / "results"

def save_data_to_folder(names, signal_times, sampling_rates, signals, labels, filename, folder_path):
    np.savez(folder_path / filename,
             names=names,
             signal_times=signal_times,
             sampling_rates=sampling_rates,
             signals=signals,
             labels=labels)


def create_window_indexes(data_size, window_size, step=1, start=0, stop=None):
    """
    Return the indexes of the windows for a given time serie size.

    :param data_size: The length of the input time serie to be cut into windows
    :param window_size: The size of each window
    :param step: The step size between each window
    :param start: The index from which to start the windows
    :param stop: The index to which to stop the windows

    :return: The indexes of all the windows
    """

    if stop is None:
        nbr_of_windows = int((data_size - (window_size-1) - start) / step)
    else:
        nbr_of_windows = int((data_size-(data_size-stop) - (window_size - 1) - start) / step)
    indexer = np.arange(start, window_size+start)[None, :] + step * np.arange(nbr_of_windows)[:, None]

    return indexer


def get_dataset_of_windows(data_signal_values, data_labels, window_size, step_between_windows, anomaly_attention_span_size, shuffle=True):
    """
    From the signal data presented as a matrix of dimension (n x d) with n being the number of signals and d the length of each signal:
    Return a matrix (NxD) of N sampled windows of size D (window_size) in the input signal along a (Nx1) vector of labels for the windows. A windows
    is associated to 1 if it contains an event in its attention span (anomaly_attention_span_size).

    :param data_signal_values: The nxd matrix of the input signals
    :param data_labels: The nx3xd cube of labels for the input signals (one hot encoding of event, amplitude, baseline of event)
    :param window_size: The desired size for the output windows
    :param step_between_windows: The step size between each windows
    :param anomaly_attention_span_size: The size of the attention inside each window (should lower than window size)
    :param shuffle: If True shuffle the output matrices

    :return: The NxD matrix of windows and the Nx1 matrix of labels
    """
    window_indexes = create_window_indexes(data_signal_values.shape[1], window_size, step=step_between_windows)

    attention_span_offset = int((window_size-anomaly_attention_span_size)/2)
    window_span_indexes = create_window_indexes(data_signal_values.shape[1], anomaly_attention_span_size, start=attention_span_offset,
                                                stop=data_signal_values.shape[1] - attention_span_offset, step=step_between_windows)

    lst_window_signal_positive = []
    lst_window_labels_positive = []
    lst_window_signal_negative = []
    lst_window_labels_negative = []
    for idx_signal in range(data_signal_values.shape[0]):
        # get all windows in the signal
        windowed_data_curr_signal = data_signal_values[idx_signal, window_indexes]
        # get all windows in the labels of the signal (narrower than windows in the data so that a window signal is labeled positive if the spike is in the center)
        windowed_labels_curr_signal = data_labels[idx_signal, :, window_span_indexes]
        # one hot vector of events in the windows
        windowed_labels_event = windowed_labels_curr_signal[:, :, 0]
        window_labels = np.max(windowed_labels_event, axis=1)
        # bool vector of windows containing an event in the attention span (narrow windows)
        window_indexes_with_1_label = window_labels != 0

        # get positive examples
        positive_labels_tmp = window_labels[window_indexes_with_1_label]
        positive_data_tmp = windowed_data_curr_signal[window_indexes_with_1_label]

        # get negative examples
        negative_labels_tmp = window_labels[np.logical_not(window_indexes_with_1_label)]
        negative_data_tmp = windowed_data_curr_signal[np.logical_not(window_indexes_with_1_label)]

        nbr_instance_to_keep = min(negative_labels_tmp.shape[0], positive_labels_tmp.shape[0])

        # build matrix of window data for the current signal

        negative_labels_indexes = np.random.permutation(negative_labels_tmp.shape[0])[:nbr_instance_to_keep]
        negative_labels = negative_labels_tmp[negative_labels_indexes]
        negative_data = negative_data_tmp[negative_labels_indexes]
        lst_window_signal_negative.append(negative_data.astype(np.float))
        lst_window_labels_negative.append(np.expand_dims(negative_labels, axis=1))

        positive_labels_indexes = np.random.permutation(positive_labels_tmp.shape[0])[:nbr_instance_to_keep]
        positive_labels = positive_labels_tmp[positive_labels_indexes]
        positive_data = positive_data_tmp[positive_labels_indexes]
        lst_window_signal_positive.append(positive_data.astype(np.float))
        lst_window_labels_positive.append(np.expand_dims(positive_labels, axis=1))

    windowed_data = np.vstack(lst_window_signal_positive + lst_window_signal_negative)
    windowed_labels = np.vstack(lst_window_labels_positive + lst_window_labels_negative)

    if shuffle:
        indexes = np.random.permutation(windowed_data.shape[0])
    else:
        indexes = np.arange(windowed_data.shape[0])
    return windowed_data[indexes], windowed_labels[indexes]

class ResultPrinter:
    """
    Class that handles 1-level dictionnaries and is able to print/write their values in a csv like format.
    """
    def __init__(self, *args, header=True, output_file=None):
        """
        :param args: the dictionnaries objects you want to print.
        :param header: tells if you want to print the header
        :param output_file: path to the outputfile. If None, no outputfile is written on ResultPrinter.print()
        """
        self.__dict = dict()
        self.__header = header
        self.__output_file = output_file

    def add(self, d):
        """
        Add dictionnary after initialisation.

        :param d: the dictionnary object you want to add.
        :return:
        """
        self.__dict.update(d)

    def _get_ordered_items(self):
        all_keys, all_values = zip(*self.__dict.items())
        arr_keys, arr_values = np.array(all_keys), np.array(all_values)
        indexes_sort = np.argsort(arr_keys)
        return list(arr_keys[indexes_sort]), list(arr_values[indexes_sort])

    def print(self):
        """
        Call this function whener you want to print/write to file the content of the dictionnaires.
        :return:
        """
        headers, values = self._get_ordered_items()
        headers = [str(h) for h in headers]
        s_headers = ",".join(headers)
        values = [str(v) for v in values]
        s_values = ",".join(values)
        if self.__header:
            print(s_headers)
        print(s_values)
        if self.__output_file is not None:
            with open(self.__output_file, "w+") as out_f:
                if self.__header:
                    out_f.write(s_headers + "\n")
                out_f.write(s_values + "\n")

class ParameterManager(dict):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--nb-filter-conv-1"] = int(self["--nb-filter-conv-1"])
        self["--nb-filter-conv-2"] = int(self["--nb-filter-conv-2"])
        self["--size-filter-conv-1"] = int(self["--size-filter-conv-1"])
        self["--size-filter-conv-2"] = int(self["--size-filter-conv-2"])
        self["--size-dense"] = int(self["--size-dense"])
        self["--learning-rate"] = float(self["--learning-rate"])
        self["--window-size"] = int(self["--window-size"])
        self["--attention-span-size"] = int(self["--attention-span-size"])
        self["--step-size"] = int(self["--step-size"])


def evaluate_labels(labels,bench,win_size):
    half_win_size=math.ceil(win_size/2)

    #compute false negative (there should be but not detected)
    FN=0
    #time points of events in the benchmark labels
    indices_bench = [i for i, x in enumerate(bench) if x == 1]
    for ind_bench in indices_bench:
        check_labels=labels[max(0,ind_bench-half_win_size):min(len(bench),ind_bench+half_win_size)]
        if 1 not in check_labels:
            FN=FN+1

    #compute false positive (there should not be but detected)
    FP=0
    #time points of events in the benchmark labels
    indices_label = [i for i, x in enumerate(labels) if x == 1]
    for ind_label in indices_label:
        check_bench=bench[max(0,ind_label-half_win_size):min(len(labels),ind_label+half_win_size)]
        if 1 not in check_bench:
            FP=FP+1

    return FN, FP
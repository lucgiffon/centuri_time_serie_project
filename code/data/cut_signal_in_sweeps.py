import numpy as np



def split_full_trace_in_sweeps(data):
    """
    Cut full trace in sweeps with the first artifact spike removed

    9 sweeps by trace
    0.1 first seconds discarded in each sweeps

    :param data: dictionnary of data.
    :return: train_sweeps_annotated, train_sweeps_not_annotated, test_sweeps
    """
    nb_sweep_by_trace = 9
    lst_idx_sweeps = np.split(np.arange(data["signals"].shape[1]), nb_sweep_by_trace)

    stacked_signal_times = np.vstack([data["signal_times"][:, idxs] for idxs in lst_idx_sweeps])
    stacked_sampling_rates = np.repeat(data["sampling_rates"], nb_sweep_by_trace)
    stacked_names = np.repeat(data["names"], nb_sweep_by_trace)
    stacked_signals = np.vstack([data["signals"][:, idxs] for idxs in lst_idx_sweeps])
    stacked_labels = np.vstack([data["labels"][:, :, idxs] for idxs in lst_idx_sweeps]) if data["labels"] is not None else None

    cutoff_time_begining = 0.1

    cutoff_indexes_begining = (cutoff_time_begining * stacked_sampling_rates).astype(np.int)

    assert np.all(cutoff_indexes_begining[0] == cutoff_indexes_begining) # needs all sampling rates to be equal

    stacked_signal_times = stacked_signal_times[:, cutoff_indexes_begining[0]:]
    stacked_signals = stacked_signals[:, cutoff_indexes_begining[0]:]
    stacked_labels = stacked_labels[:, :, cutoff_indexes_begining[0]:] if stacked_labels is not None else None
    # no change in stack sampling rates nor names


    out_data = {}
    out_data["signal_times"] = stacked_signal_times
    out_data["sampling_rates"] = stacked_sampling_rates
    out_data["signals"] = stacked_signals
    out_data["labels"] = stacked_labels
    out_data["names"] = stacked_names

    return out_data


def discard_data_with_no_annotation(data):
    names = data["names"]
    sampling_rates = data["sampling_rates"]
    signals = data["signals"]
    labels = data["labels"]
    signal_times = data["signal_times"]

    presence_of_annotation_signal = np.max(labels[:, 0, :], axis=1).astype(np.bool)

    kept_data = {
        "signal_times": signal_times[presence_of_annotation_signal],
        "labels": labels[presence_of_annotation_signal],
        "signals": signals[presence_of_annotation_signal],
        "sampling_rates": sampling_rates[presence_of_annotation_signal],
        "names": names[presence_of_annotation_signal]
    }
    discarded_data = {
        "signal_times": signal_times[np.logical_not(presence_of_annotation_signal)],
        "labels": None,
        "signals": signals[np.logical_not(presence_of_annotation_signal)],
        "sampling_rates": sampling_rates[np.logical_not(presence_of_annotation_signal)],
        "names": names[np.logical_not(presence_of_annotation_signal)]
    }

    return kept_data, discarded_data

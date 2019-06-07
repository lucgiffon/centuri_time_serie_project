from centuri_project.utils import processed_directory
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_file = processed_directory / "splitted_train_with_annotations.npz"
    data = np.load(data_file, allow_pickle=True)
    data_sweepX = data["signal_times"]  # for all trace the time for each value record (n x d matrix)
    data_sampling_rates = data["sampling_rates"]  # for all trace the sampling rate in Hz (n sized vector)
    data_signal_values = data["signals"]  # for all trace the value records (n x d matrix): the actual traces we need to deal with
    data_labels = data["labels"]  # for all trace, the event presence (as 1/0 vector), the event amplitude and the baseline (n x 3 x d cube of data)
    data_names = data["names"]

    for idx_sweep, _ in enumerate(data_signal_values):
        if idx_sweep > 5: # shows only the 5 first sweeps of all the dataset
            break
        trace_y = data_signal_values[idx_sweep]
        trace_x = data_sweepX[idx_sweep]
        sampling_rate_trace = data_sampling_rates[idx_sweep]
        labels = data_labels[idx_sweep]
        onehot_labels = labels[0].astype(np.bool)
        amplitude_labels = labels[1]
        baseline_labels = labels[2]
        real_amplitudes = baseline_labels - amplitude_labels
        name = data_names[idx_sweep]

        cutedoff_trace_x = trace_x
        cutedoff_trace_y = trace_y

        f, ax = plt.subplots()

        ax.plot(cutedoff_trace_x, cutedoff_trace_y, color="c", zorder=-2, label="raw data")
        ax.scatter(cutedoff_trace_x[onehot_labels], cutedoff_trace_y[onehot_labels], edgecolors="r", facecolors="none", s=80, zorder=1, label="event re-position")
        ax.scatter(cutedoff_trace_x[onehot_labels], real_amplitudes[onehot_labels], edgecolors="b", facecolors="none", s=80, zorder=1, label="event annotated")
        ax.scatter(cutedoff_trace_x[onehot_labels], baseline_labels[onehot_labels], marker="x", color="y", s=80, zorder=1, label="baseline")

        plt.title(name)
        plt.legend()
        plt.show()

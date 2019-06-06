from centuri_project.utils import processed_directory
import numpy as np

if __name__ == "__main__":
    data_file = processed_directory / "train.npz"
    data = np.load(data_file, allow_pickle=True)
    data_sweepX = data["signal_times"]  # for all trace the time for each value record (n x d matrix)
    data_sampling_rates = data["sampling_rates"]  # for all trace the sampling rate in Hz (n sized vector)
    data_signal_values = data["signals"]  # for all trace the value records (n x d matrix): the actual traces we need to deal with
    data_labels = data["labels"]  # for all trace, the event presence (as 1/0 vector), the event amplitude and the baseline (n x 3 x d cube of data)

    # here happens the magic
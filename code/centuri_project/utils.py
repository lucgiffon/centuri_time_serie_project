import pathlib
import numpy as np


project_dir = pathlib.Path(__file__).resolve().parents[2]

raw_data_directory = project_dir / pathlib.Path("data/raw/1_time_series_analysis/EPSC/Traces & Events")
interim_directory = project_dir / pathlib.Path("data/interim")
processed_directory = project_dir / pathlib.Path("data/processed")


def save_data_to_folder(names, signal_times, sampling_rates, signals, labels, filename, folder_path):
    np.savez(folder_path / filename,
             names=names,
             signal_times=signal_times,
             sampling_rates=sampling_rates,
             signals=signals,
             labels=labels)

from centuri_project.utils import save_data_to_folder, interim_directory, processed_directory
from cut_signal_in_sweeps import split_full_trace_in_sweeps, discard_data_with_no_annotation
from filter_signal_files import get_filtered_signals

if __name__ == "__main__":
    filtered_train_data, filtered_test_data = get_filtered_signals()

    save_data_to_folder(**filtered_train_data, filename="filtered_train", folder_path=interim_directory)
    save_data_to_folder(**filtered_test_data, filename="filtered_test", folder_path=interim_directory)

    splitted_train_data, splitted_test_data = split_full_trace_in_sweeps(filtered_train_data), split_full_trace_in_sweeps(filtered_test_data)

    kept_train_data, discarded_train_data = discard_data_with_no_annotation(splitted_train_data)

    save_data_to_folder(**kept_train_data, filename="splitted_train_with_annotations", folder_path=processed_directory)
    save_data_to_folder(**discarded_train_data, filename="splitted_train_without_annotations", folder_path=processed_directory)
    save_data_to_folder(**splitted_test_data, filename="splitted_test", folder_path=processed_directory)

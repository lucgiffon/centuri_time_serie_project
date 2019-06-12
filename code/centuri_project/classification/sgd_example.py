from scipy.signal import detrend
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from centuri_project.utils import processed_directory, get_dataset_of_windows
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


if __name__ == "__main__":
    data_file = processed_directory / "splitted_train_with_annotations.npz"  # all train data with annotations
    data = np.load(data_file, allow_pickle=True)
    data_signal_values = data["signals"]  # for all trace the value records (n x d matrix): the actual traces we need to deal with
    data_labels = data["labels"]  # for all trace, the event presence (as 1/0 vector), the event amplitude and the baseline (n x 3 x d cube of data)
    data_names = data["names"]

    mapping = {}
    for i in np.unique(data_names):
        mapping[i] = np.where(data_names == i)[0]

    test_names = np.random.choice(list(mapping.keys()), 2, replace=False)

    # remove a portion of labeled samples so that we can later evaluate event detection
    test_indices = [item for sublist in [mapping[tst_na] for tst_na in test_names] for item in sublist]
    data_signal_values_test = data_signal_values[test_indices]
    data_labels_test = data_labels[test_indices]

    train_indices = list(set(range(data_signal_values.shape[0])) - set(test_indices))
    data_signal_values = data_signal_values[train_indices]
    data_labels = data_labels[train_indices]

    window_size = 500  # The size of each window (signal cut)
    attention_span = 200  # The attention span in which to look for events in each window
    # |   <--attention span-->   |
    # |<-------window size------>|
    step_between_windows = 20  # the step size between each first index of windows
    X, y = get_dataset_of_windows(data_signal_values, data_labels, window_size, step_between_windows, attention_span)
    X = normalize(X, axis=1)  # highly recommended
    X = detrend(X)  # make the signals trend horizontal

    # train val split is necessary to evaluate classification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

    # visualize some examples
    for i in range(5):
        plt.plot(X[i])
        plt.title(y[i])
        plt.show()

    # simple classification example; you should replace this with your own machine learning process
    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(X_train, y_train)
    print(clf.score(X_val, y_val))
"""
Analysis of objective function during qmeans execution

Usage:
  cnn_example [-h] --nb-filter-conv-1=int --nb-filter-conv-2=int --size-filter-conv-1=int --size-filter-conv-2=int --size-dense=int --learning-rate=float --window-size=int --attention-span-size=int --step-size=int

Options:
  -h --help                             Show this screen.
  -v --verbose                          Set verbosity to debug.
  --seed=int                            The seed to use for numpy random module.

CNN architecture:
  --nb-filter-conv-1=int                Number of filters in the first convolution.
  --nb-filter-conv-2=int                Number of filters in the second convolution.
  --size-filter-conv-1=int              Size of filters in the first convolution.
  --size-filter-conv-2=int              Size of filters in the second convolution.
  --size-dense=int                      Size of the intermediate dense layer.
  --learning-rate=float                 Learning rate.

Data:
  --window-size=int                     Size of the windows cut in the data.
  --attention-span-size=int             Size of the attention span in each window.
  --step-size=int                       Step between each window.

"""

import time
import sys
import docopt

import keras
from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy.signal import detrend
from sklearn.model_selection import train_test_split

from centuri_project.utils import get_dataset_of_windows, project_dir, ParameterManager, ResultPrinter, prepare_train_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

lst_results_header = [
    "traintime",
    "val_loss",
    "val_acc",
    "loss",
    "acc",
]

if __name__ == "__main__":
    print("Command line: " + " ".join(sys.argv))
    output_models_path = project_dir / "models/trained/classification/cnn"
    output_results_path = project_dir / "results/classification/cnn"
    identifier = str(int(time.time()))

    out_param = {
        "output_models_path": output_models_path,
        "output_results_path": output_results_path,
        "identifier": identifier
    }

    arguments = docopt.docopt(__doc__)
    paraman = ParameterManager(arguments)
    initialized_results = dict((v, None) for v in lst_results_header)
    resprinter = ResultPrinter(output_file=output_results_path / (identifier + ".csv"))
    resprinter.add(out_param)
    resprinter.add(initialized_results)
    resprinter.add(paraman)

    try:

        data_signal_values, data_labels = prepare_train_data()

        window_size = paraman["--window-size"]  # The size of each window (signal cut)
        attention_span = paraman["--attention-span-size"]  # The attention span in which to look for events in each window
        # |   <--attention span-->   |
        # |<-------window size------>|
        step_between_windows = paraman["--step-size"]  # the step size between each first index of windows

        if attention_span > window_size:
            raise ValueError("Attention span {} is greater than window size {}".format(attention_span, window_size))

        X, y = get_dataset_of_windows(data_signal_values, data_labels, window_size, step_between_windows, attention_span)
        X = normalize(X, axis=1)  # highly recommended
        X = detrend(X)  # make the signals trend horizontal
        X = np.expand_dims(X, axis=-1)
        y = to_categorical(y)

        # train val split is necessary to evaluate classification
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

        # visualize some examples
        for i in range(5):
            plt.plot(X[i])
            plt.title(y[i])
            plt.show()

        # CNN model definition
        ######################
        inputs = Input(shape=(window_size,1))
        rep = Conv1D(paraman["--nb-filter-conv-1"], paraman["--size-filter-conv-1"])(inputs)
        rep = MaxPooling1D(3)(rep)
        rep = Conv1D(paraman["--nb-filter-conv-2"], paraman["--size-filter-conv-2"])(rep)
        rep = MaxPooling1D(3)(rep)
        rep = Flatten()(rep)
        rep = Dropout(rate=0.5)(rep)
        rep = Dense(paraman["--size-dense"], activation='relu')(rep)
        rep = Dropout(rate=0.5)(rep)
        predictions = Dense(2, activation='softmax')(rep)
        model = Model(inputs=[inputs], outputs=predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(paraman["--learning-rate"]), metrics=['accuracy'])


        print(model.summary())

        callbacks_list = [
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]


        BATCH_SIZE = 128
        EPOCHS = 20

        start = time.time()
        history = model.fit(X_train,
                            y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            callbacks=callbacks_list,
                            validation_split=0.2,
                            verbose=1)
        stop = time.time()
        traintime = stop - start

        res = {
            "traintime": traintime,
              "val_loss": history.history["val_loss"][-1],
              "val_acc": history.history["val_acc"][-1],
              "loss": history.history["loss"][-1],
              "acc": history.history["acc"][-1],
        }
        resprinter.add(res)
        # resprinter.add(history.history)
        model.save((output_models_path / ("cnn_{}.h5".format(identifier))).absolute().as_posix())
    finally:
        resprinter.print()
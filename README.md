# Setting up the project

## Installation

If you can, use a new conda environment for this project. Python 3 is necessary:

	conda create -n centuri_1 python=3
	source activate centuri

Go to the `/code` directory then type:

	pip install -e .

This will install the required dependencies and the project files.

## Prepare data

Move the directory `1 - Time Series Analysis` provided by the tutor Davide to the `data/raw` directory
and rename it `1_time_series_analysis`.

Go to the `/code/data` directory then execute the file `filter_signal_files.py` then the file `cut_signal_in_sweeps.py`.

	python process_data.py
	
To make sure everything has been done properly, go to the `/code/vizualiation` directory then execute the `show_data_sweeps.py` file.

	python show_data_sweeps.py

If everything is fine, there should be no error and few plots should show up (around 5).

# Start coding

If you want to code something new for the project. 
Go to the `/code/centuri_project` directory and look at the file `load_data_example.py` to see how to load the data.
All your new code files must go under the `/code/centuri_project` directory.

## Play with windows of signal

If you want to play with labeled windows of signal, check out the script `code/centuri_project/classification/sgd_example.py` to see 
how to load the windows from the signal data files.
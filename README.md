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

	python filter_signal_files.py
	python cut_signal_in_sweeps.py
	
To make sure everything has been done properly, go to the `/code/vizualiation` directory then execute the `show_data_sweeps.py` file.

	python show_data_sweeps.py

If there is no error, everything is fine.

# Start coding

If you want to code something new for the project. 
Go to the `/code/centuri_project` directory and look at the file `load_data_example.py` to see how to load the data.
All your new code files must go under the `/code/centuri_project` directory.

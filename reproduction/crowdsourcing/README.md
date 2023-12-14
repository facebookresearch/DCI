# Crowdsourcing

This directory includes the tasks used to collect the dataset, as well as the scripts required to prepare and package the resultant data.

- **`annotate`**: Contains the main task we use for data collection.
- **`qualify`**: Contains a screening task used to primarily filter out bots and workers who wouldn't be able to complete the main task.
- **`prepare`**: Scripts to prepare the dataset for release, following it being collected. 

Both tasks require the [mephisto](https://github.com/facebookresearch/Mephisto) library to use (v1.2). You should clone it into a directory and `pip install -e .`.
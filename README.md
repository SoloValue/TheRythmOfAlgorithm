# Image retrieval (similarity) model

This model works as an image search engine that takes a query image as input and returns the N most similar images from a gallery. It works with convolutional neural networks and transfer learning.


## Contents
- `comp_data`: folder containing gallery and query images from competition day.
- `config`: folder containing a yaml file for configuration of models, training, and directories.
- `json_files`: folder containing the following .json files:
    - `runs_recap.json`: JSON file with training performances of models.
    - `comp_results.json`: JSON file where final results from the competition are saved (model name, top-k accuracy, output of the model and distances of similar images).
- `test_gallery`: folder containing gallery and query used to compute `test_error` during training.
- `train_gallery`: folder containing images used for training.
- `competition.py`: working file for the competition. The model takes query and gallery contained in `comp_data` as inputs and submits and saves the output.
- `dataset.py`: defines dataset and dataloader objects.
- `main.py`: executes training (with synchronization on wandb).
- `model_selection.py`: executes training of different models and finds the best performing.
- `network.py`: defines the architecture of all models (networks).
- `plot_results.py`: plots and saves final results.
- `plot_runs.py`: plots performances from `model_selection.py` and `runs_recap.json`.
- `query.py`: simulation of the competition.
- `trainer.py`: defines train, validation, test and competition steps.
- `utils.py`: contains helper functions.

## Authors

- Elisa Basso
- Marta Faggian [@martafaggian](https://github.com/martafaggian)
- Matteo Moscatelli
- Sara Tegoni [@sraatgn](https://github.com/sraatgn)


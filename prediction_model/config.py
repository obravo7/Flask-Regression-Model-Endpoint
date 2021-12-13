from pathlib import Path

BASE_PATH = Path(__file__).absolute().parent

"""
TODO: It is better to create a function called load_model() that takes in the trained model path as an argument.
Load Model config based on `parameters.json`
"""


class ModelConfig:
    model_path = BASE_PATH / "docs" / 'model' / 'best-model.pt'

    learning_rate = 0.007
    n_estimators = 10000
    max_depth = 8
    min_child_weight = 1.0
    subsample = 0.6
    reg_alpha = 0.75
    reg_lambda = 0.45
    gamma = 0
    n_jobs = 16

import xgboost as xgb
import pandas as pd
import numpy as np

from pathlib import Path

try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig

try:
    from .dataset.load import TrainData
except ImportError:
    from dataset.load import TrainData

BASE_PATH = Path.cwd().parent
TEST_CSV_PATH = BASE_PATH / "data" / "mercari_test.csv"


class Model:
    def __init__(self,
                 model_path,
                 learning_rate,
                 n_estimators,
                 max_depth,
                 min_child_weight,
                 subsample,
                 reg_alpha,
                 reg_lambda,
                 gamma,
                 n_jobs
                 ):
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_jobs = n_jobs

        self.reg = xgb.XGBRegressor(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            subsample=self.subsample,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            n_jobs=self.n_jobs
        )

        self.reg.load_model(model_path)

    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(data_df)


def predict_price(data: pd.DataFrame):
    reg = Model(model_path=ModelConfig.model_path,
                learning_rate=ModelConfig.learning_rate,
                n_estimators=ModelConfig.n_estimators,
                max_depth=ModelConfig.max_depth,
                subsample=ModelConfig.subsample,
                reg_alpha=ModelConfig.reg_alpha,
                reg_lambda=ModelConfig.reg_lambda,
                gamma=ModelConfig.gamma,
                n_jobs=ModelConfig.n_jobs,
                min_child_weight=ModelConfig.min_child_weight)

    res = reg.predict(data)
    return res


def main():
    # create submission file
    td = TrainData(TEST_CSV_PATH)
    id_list = td.data['id'].to_list()
    test_data = td.get_test_data()

    prices = predict_price(test_data)

    pd.DataFrame({'id': id_list, 'price': [int(price) for price in prices]}).to_csv('results.csv', index=False)


if __name__ == "__main__":
    main()

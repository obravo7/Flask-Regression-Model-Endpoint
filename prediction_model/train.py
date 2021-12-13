import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import json
from pathlib import Path
from argparse import ArgumentParser

from dataset.load import TrainData

BASE_PATH = Path.cwd().parent
TRAIN_CSV_PATH = BASE_PATH / "data" / "mercari_train.csv"
TEST_CSV_PATH = BASE_PATH / "data" / "mercari_test.csv"
OUTPUT_BASE_DIR = BASE_PATH / "prediction_model" / "results"
CATEGORIES_JSON_PATH = BASE_PATH / "prediction_model" / "dataset" / "categories.json"
BRANDS_JSON_PATH = BASE_PATH / "prediction_model" / "dataset" / "brand_category.json"


class Train:
    def __init__(self, train_csv: Path, verbose=True):
        self.train_csv = train_csv
        self.train_data = TrainData(self.train_csv)

        self.parameters = None  # save parameters from training run
        self.reg = None
        self.base_save_path = None
        self.verbose = verbose

    def create_one_hot_files(self, file_name, column_name: str):
        self.train_data.create_onehot_mapping(file_name, column_name)

    def run(self,
            evaluation_metrics: list,
            learning_rate: float,
            n_estimators: int,
            max_depth: int,
            min_child_weight: float,
            subsample: float,
            reg_alpha: float,
            reg_lambda: float,
            gamma: int,
            n_jobs: int,
            base_save_path: Path,
            early_stopping_rounds: int,
            save_training_images: bool = False,
            ):

        x_train, y_train, x_eval, y_eval = self.train_data.split_train_data()

        self._check_and_print(
            message=self._message_template(
                message=' ||  '.join(list(x_train.columns)) + f"\n{x_train.head}",
                title="Training Matrix Head"
            )
        )
        self._check_and_print(message=self._message_template(message=f'{y_train.head}', title="Target Vector Head"))

        self.parameters = {
            "evaluation_metrics": evaluation_metrics,
            "learning_rate": learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'gamma': gamma,
            'n_jobs': n_jobs,
            # 'base_save_path': base_save_path.__str__(),
            'early_stopping_rounds': early_stopping_rounds,
            'save_training_images': save_training_images
        }

        # print: dict to str
        dict_msg = ",  \n".join('{}={}'.format(*t) for t in zip(self.parameters.keys(), self.parameters.values()))
        self._check_and_print(message=self._message_template(dict_msg, title="Training Parameters"))
        self.base_save_path = base_save_path

        # setup + run training
        self.reg = xgb.XGBRegressor(learning_rate=learning_rate,
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_child_weight=min_child_weight,
                                    subsample=subsample,
                                    reg_alpha=reg_alpha,
                                    reg_lambda=reg_lambda,
                                    gamma=gamma,
                                    n_jobs=n_jobs,
                                    )

        self.reg.fit(x_train, y_train,
                     eval_set=[(x_train, y_train), (x_eval, y_eval)],
                     eval_metric=evaluation_metrics,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose=self.verbose)

        # store training loss plots
        evals_result: dict = self.reg.evals_result()
        if save_training_images:
            matplotlib.rc('font', family='TakaoPGothic')  # display Japanese text if needed
            ax = xgb.plot_importance(self.reg, ax=None)
            ax.figure.savefig(base_save_path / f"importance.png")

            epochs = len(evals_result['validation_0']['rmse'])
            x_axis = range(0, epochs)
            for metric in evaluation_metrics:

                # plot epochs vs metric loss
                fig, ax = plt.subplots(figsize=(16, 12))
                self.parameters[metric] = {
                    'train': evals_result['validation_0'][metric][-1],
                    'validation': evals_result['validation_1'][metric][-1],
                }
                ax.plot(x_axis, evals_result['validation_0'][metric], label='Train')
                ax.plot(x_axis, evals_result['validation_1'][metric], label='Validation')
                ax.legend()
                plt.xlabel("epochs")
                plt.ylabel(metric.upper())
                plt.title(f'XGBoost Regression {metric.upper()}')
                ax.figure.savefig(base_save_path / f"training-{metric.upper()}.png")

        # store training files
        self.reg.save_model(base_save_path / f'best-model.pt')

        parameter_file = base_save_path / f'parameters.json'
        with open(parameter_file, "w") as out:
            json.dump(self.parameters, out, ensure_ascii=False, indent=4)

        history_file = base_save_path / f'history.json'
        with open(history_file, "w") as out:
            json.dump(evals_result, out, ensure_ascii=False, indent=4)

    def grid_search(self):
        pass  # todo

    def create_results(self, test_csv_path: Path):
        """
        Create `results.csv`, a file containing the prediction results for the test set. The columns
        are ID

        :param test_csv_path: (pathlib.Path) Path to test csv file.
        :return: None
        """
        if not self.reg:
            raise ValueError('Model not trained! Please run Train.run(...) to initialize model')
        td = TrainData(test_csv_path)
        id_list = td.data['id'].to_list()
        test_data = td.get_test_data()

        predicted_price = self.reg.predict(test_data)
        pd.DataFrame(
            {'id': id_list, 'price': [int(price) for price in predicted_price]}
        ).to_csv(
            self.base_save_path / 'results.csv',
            index=False
        )

    def _check_and_print(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    @staticmethod
    def _print_lines(n) -> str:
        return f"{'='*n}"

    def _message_template(self, message: str, title: str = '', line_length: int = 50) -> str:
        return f"{self._print_lines(line_length)}\n" \
               f"{title}\n" \
               f"{self._print_lines(line_length)}\n" \
               f"{message}\n\n" \
               f"{self._print_lines(line_length)}\n\n"

    @staticmethod
    def callback():
        # todo: add callback function that logs XGBoost stdout log
        pass


def parse_arguments():
    parser = ArgumentParser()

    # files
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--base-output-dir", default=OUTPUT_BASE_DIR)
    parser.add_argument("--train-csv", default=TRAIN_CSV_PATH)
    parser.add_argument('--test-csv', default=TEST_CSV_PATH)
    parser.add_argument("--categories-json-path", default=CATEGORIES_JSON_PATH)
    parser.add_argument("--brands-json-path", default=BRANDS_JSON_PATH)
    parser.add_argument("--create-onehot-files", default=False, type=bool)
    parser.add_argument("--save-training-images", default=True, type=bool)
    parser.add_argument("--verbose", default=True, type=bool)

    # model parameters
    # parser.add_argument("--grid-search", default=False, type=bool)
    parser.add_argument("--learning-rate", default=0.007, type=float)
    parser.add_argument("--early-stopping-rounds", default=50, type=int)
    parser.add_argument("--n-estimators", default=10000, type=int)

    parser.add_argument("--max-depth", default=8, type=int)
    parser.add_argument("--min-child-weight", default=1.0, type=float)

    parser.add_argument("--subsample", default=0.6, type=float)
    parser.add_argument("--reg-alpha", default=0.75, type=float)
    parser.add_argument("--reg-lambda", default=0.45, type=float)
    parser.add_argument("--gamma", default=0, type=int)
    parser.add_argument("--n-jobs", default=16, type=int)

    # metrics
    parser.add_argument("--rmse", default=True, type=bool)
    parser.add_argument("--rmsle", default=True, type=bool)
    parser.add_argument("--mae", default=True, type=bool)

    args = parser.parse_args()

    evaluation_metrics = []
    if args.rmsle:
        evaluation_metrics.append('rmsle')
    if args.rmse:
        evaluation_metrics.append('rmse')
    if args.mae:
        evaluation_metrics.append('mae')

    args.evaluation_metrics = evaluation_metrics

    args.save_folder = args.base_output_dir / args.out_dir
    args.save_folder.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_arguments()

    train = Train(args.train_csv, args.verbose)
    if args.create_onehot_files:
        train.create_one_hot_files('category_name', args.categories_json_path)
        train.create_one_hot_files('brand_name', args.brands_json_path)

    train.run(evaluation_metrics=args.evaluation_metrics,
              learning_rate=args.learning_rate,
              n_estimators=args.n_estimators,
              max_depth=args.max_depth,
              min_child_weight=args.min_child_weight,
              subsample=args.subsample,
              reg_alpha=args.reg_alpha,
              reg_lambda=args.reg_lambda,
              gamma=args.gamma,
              n_jobs=args.n_jobs,
              base_save_path=args.save_folder,
              early_stopping_rounds=args.early_stopping_rounds,
              save_training_images=args.save_training_images,
              )

    train.create_results(args.test_csv)


if __name__ == "__main__":
    main()

# from dataset.encoding import OneHot
from .encoding import OneHot

import pandas as pd
import numpy as np

from typing import Tuple, Union, Optional, List
import json
from pathlib import Path


class DataGenerator:
    def __init__(self):
        pass

    def __next__(self):
        pass

    def __len__(self):
        pass


class TrainData:

    def __init__(self,
                 file_path: Union[Path, pd.DataFrame],
                 drop_columns: Optional[List[str]] = None,
                 target_column: Optional[str] = None):

        if isinstance(file_path, Path):
            self.data = pd.read_csv(file_path)
        elif isinstance(file_path, pd.DataFrame):
            self.data = file_path
        else:
            raise ValueError(f'Unrecognized file type for argument "file_path", with: {type(file_path)}')

        # manage target and unnecassary columns
        self.drop_columns = drop_columns
        if not self.drop_columns:
            self.drop_columns = ['item_description', 'name', 'id']
        self.target_column = target_column
        if not self.target_column:
            self.target_column = 'price'

    def split_train_data(self, train_split: float = 0.8
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # shuffle data
        np.random.seed(None)
        perm = np.random.permutation(self.data.index)
        m = len(self.data.index)
        train_end = int(train_split * m)

        # copy, and one-hot conversion
        data = self.data.copy(deep=True)
        category_row = OneHot.categories(list(data['category_name']))
        brands_row = OneHot.brands(list(data['brand_name']))
        data['category_name'] = category_row
        data['brand_name'] = brands_row

        train_data = data.iloc[perm[:train_end]]
        validate_data = data.iloc[perm[train_end:]]

        train_data_target = train_data[self.target_column]
        train_data = train_data.drop([self.target_column] + self.drop_columns, axis=1)

        validate_data_target = validate_data[self.target_column]
        validate_data = validate_data.drop([self.target_column] + self.drop_columns, axis=1)

        return train_data, train_data_target, validate_data, validate_data_target

    def get_test_data(self):
        data = self.data.copy(deep=True)
        category_row = OneHot.categories(list(data['category_name']))
        brands_row = OneHot.brands(list(data['brand_name']))

        # for brand in brands_row:
        #     if isinstance(brand, str):
        #         print(f'brand={brand}')

        data['category_name'] = category_row
        data['brand_name'] = brands_row

        test_data = data.drop(self.drop_columns, axis=1)
        return test_data

    def k_folds_data(self) -> DataGenerator:
        pass

    def create_onehot_mapping(self, file_name, column_name='category_name'):
        categories = self.data[column_name].to_list()

        unique_categories = np.unique(categories)
        print(f'number of categories: {len(unique_categories)}')

        category_value = 0
        category_mapping = {}
        print(f'')
        for category in unique_categories:
            category_mapping[category] = category_value
            category_value += 1

        with open(file_name, "w") as out:
            json.dump(category_mapping, out, ensure_ascii=False, indent=4)

        return file_name





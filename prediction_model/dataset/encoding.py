import json
from pathlib import Path


BASE_PATH = Path(__file__).absolute().parent.parent
BRAND_CATEGORIES_JSON_PATH = BASE_PATH / "dataset" / "brand_category.json"
CATEGORIES_JSON_PATH = BASE_PATH / "dataset" / "categories.json"


class OneHot:
    @classmethod
    def categories(cls, data: list, matching=None):
        if matching is None:
            with open(CATEGORIES_JSON_PATH) as f:
                matching = json.load(f)
        return convert_one_hot(data, matching)

    @classmethod
    def brands(cls, data: list, matching=None):
        if matching is None:
            with open(BRAND_CATEGORIES_JSON_PATH) as f:
                matching = json.load(f)
        return convert_one_hot(data, matching)

    @classmethod
    def names(cls, data: list, matching: dict):
        if matching is None:
            raise NotImplementedError  # not required
        return convert_one_hot(data, matching)


def convert_one_hot(data, matching):
    """
    Converts items in data using the matching dictionary. If a value in data is not in
    matching, then float('nan') is used instead (if not already nan value).
    """
    for idx, val in enumerate(data):
        if val in matching:
            data[idx] = matching[val]
        elif isinstance(val, str):
            data[idx] = float('nan')
    return data



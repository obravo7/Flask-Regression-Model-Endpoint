import unittest
import json
from flask import Flask
from site_apis.price_api import prediction_api

app = Flask(__name__)
app.register_blueprint(prediction_api)


class PricePredictionTests(unittest.TestCase):

    tester = None

    def __init__(self, *args, **kwargs):
        super(PricePredictionTests, self).__init__(*args, **kwargs)
        global tester
        tester = app.test_client()

    def test_all_nan(self):
        response = tester.get(
            '/v1/price',
            data=json.dumps(
                {}
            ),
            content_type="application/json"
        )

        data = response.get_data(as_test=True)
        self.assertEqual(response.status, 200)
        self.assertIsNotNone(data)

    def test_invalid_types(self):
        pass


if __name__ == "__main__":
    unittest.main()

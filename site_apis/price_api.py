from flask import jsonify, request, Blueprint


from prediction_model.dataset.load import TrainData
from prediction_model.predict import predict_price

import pandas as pd

prediction_api = Blueprint('prediction_api', __name__)


@prediction_api.route('/v1/price', methods=['POST', "GET"])
def price():

    default = None
    id_value = request.args.get('id', default=default, type=int)
    name = request.args.get('name', default=default, type=str)
    item_condition = request.args.get('item_condition_id', default=default, type=str)
    category_name = request.args.get('category_name', default=default, type=str)
    brand_name = request.args.get('brand_name', default=default, type=str)
    shipping = request.args.get('shipping', default=default, type=int)
    item_description = request.args.get('item_description', default=default, type=str)
    seller_id = request.args.get('seller_id', default=default, type=int)

    args = {'id': [id_value if id_value else float('nan')],
            "name": [name if name else float('nan')],
            'item_condition_id': [item_condition if item_condition else float('nan')],
            'category_name': [category_name if category_name else float('nan')],
            'brand_name': [brand_name if brand_name else float('nan')],
            'shipping': [shipping if shipping else float('nan')],
            'seller_id': [seller_id if seller_id else float('nan')],
            'item_description': [item_description if item_description else float('nan')]
            }
    args_df = pd.DataFrame(args)
    td = TrainData(args_df)
    data = td.get_test_data()
    predicted_price = predict_price(data)

    return jsonify(price=int(predicted_price[0])), 200

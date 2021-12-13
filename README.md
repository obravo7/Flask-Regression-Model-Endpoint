#  Price Prediction API

The purpose of this repo is to develop a price prediction model and a 
corresponding REST endpoint. Why? To learn, mostly.

---
## Run API Development Server

A Dockerfile is provided to run a local development server. Use Python 3.7. For dependencies, see `requirements.txt`.

Build Docker image container
```
docker build . -t price-app
```

Run locally: 
```
docker run -p 80:80 price-app
```

### Unit tests for the API server

A basic unit test for the Endpoint can be found in `test_price_api.py`. 

### Response

After the development server is running, navigate to [`http://localhost/v1/price`](http://localhost/v1/price). 

Since the prediction model takes in sparse data, it is possible to fetch the endpoint without any arguments. This should yield:

```
{
  'price': 30
}
```
### Request body

- Takes any information listed in the `data/` directory.

---
## Training Code and Model Info

Please see `prediction_model/README.md` for statistical model info, as well as training plots and
results.

### Prediction Model
Trained model can be found in `prediction_model/docs/model/best-model.pt`. The size of the file is `4.5MB`.

---
## Results

The model provided in this repo received a RMSLE score of `0.582849` during training, and a RMSLE score of `0.610435` in the validation set.
The training results are as follows:


| Metric | Train | Validation  |
| --- | --- | --- | 
| `RMSLE` | `0.586161` | `0.607384` |
| `RMSE` | `20.846642` | `25.111233` |
| `MAE` | `10.420339` | `11.088115` |


### Prediction results
The predicted price values for the `data/mercari_test.csv` file can be found in `results.csv`.
See `prediction_model/README.md` for more details regarding the prediction model. 
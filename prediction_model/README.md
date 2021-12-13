# Price Prediction Model

This repo contains a trained model for price prediction. The predictor is a regression model trained
using [XGBoost](https://xgboost.readthedocs.io/en/stable/) --a distributed gradient boosting library. 

---
## Training

To train the model using the parameters used in `docs/model/best-model.pt`, run the command:       
```python
python train.py --out-dir "FolderName"
```

###Available training parameters for `train.py`:
 - `--learning-rate`: Starting learning rate. Defaults=`0.007`.
 - `--early-stopping-rounds`: The model will train until the validation score stops improving. Validation error needs to decrease at least every `early_stopping_rounds` to continue training. Defaults=`50`.
 - `--n-estimators`:  The number of trees in the ensemble. Equivalent to the number of number boosting rounds. Default=`10000`.
 - `--max-depth`: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. Default=`8`.
 - `--min-child-weight`: Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be. Default=1.0
 - `--subsample`: Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Default=0.6
 - `--reg-lambda`:  L2 regularization term on weights. Increasing this value will make model more conservative. Default=`0.75` 
 - `--reg-alpha`: L1 regularization term on weights. Increasing this value will make model more conservative. Default=`0.45` 
 - `--gamma`: Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. Default=0

For more arguments please see `train.py`. For more info on these parameters, please see the [Official Documentation](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster).

###Output Results


Results are stored in `results/FolderName`, where `FolderName` is the name of the folder passed to `train.py`. The files include:
 - `best-model`: trained model
 - `history.json`: training history
 - `parameters.json`: parameters used in model.
 - `results.csv`: contains predicted prices for `mercari-test.csv`. 
 - `importance.png`: Feature importance plot.
 - `training-RMSLE.png`: RMSLE loss plot. 
 - `training-RMSE.png`: RMSE loss plot.
 - `training-MAE.png`: MAE loss plot. 


The run parameters are stored inside the `out-dir` directory as a dictionary, and contain the models evaluation scores. 

Example of `parameters.json`:

```python
{{
    "evaluation_metrics": [
        "rmsle",
        "rmse",
        "mae"
    ],
    "learning_rate": 0.007,
    "n_estimators": 10000,
    "max_depth": 8,
    "min_child_weight": 1.0,
    "subsample": 0.6,
    "reg_alpha": 0.75,
    "reg_lambda": 0.45,
    "gamma": 0,
    "n_jobs": 16,
    "early_stopping_rounds": 50,
    "save_training_images": true,
    "rmsle": {
        "train": 0.586161,
        "validation": 0.607384
    },
    "rmse": {
        "train": 20.846642,
        "validation": 25.111233
    },
    "mae": {
        "train": 10.420339,
        "validation": 11.088115
    }
}
```
---
## Results

Results corresponding to `docs/models/best-model.pt`

### Metrics

| Metric | Train | Validation  |
| --- | --- | --- | 
| `RMSLE` | `0.586161` | `0.607384` |
| `RMSE` | `20.846642` | `25.111233` |
| `MAE` | `10.420339` | `11.088115` |


### Plots

| RMSLE | RMSE | MAE  |
| --- | --- | --- | 
| <img src="https://github.com/m-rec/f53dca0e7768853885081d752cdded81086d4434/blob/master/prediction_model/docs/training-RMSLE.png" height="250">| <img src="https://github.com/m-rec/f53dca0e7768853885081d752cdded81086d4434/blob/master/prediction_model/docs/training-RMSE.png" height="250">| <img src="https://github.com/m-rec/f53dca0e7768853885081d752cdded81086d4434/blob/master/prediction_model/docs/training-MAE.png" height="250"> |


#### Importance

<img src="https://github.com/m-rec/f53dca0e7768853885081d752cdded81086d4434/blob/master/prediction_model/docs/importance.png" height="250">

According to the model, the features that are most important are the `seller_id` and the `brand_name`. The brand name
should be an obvious indicator of price. If the `seller_id` corresponds to the ID of the store in which the item was sold, then
this is telling us that the location of where the item was sold might also be important.
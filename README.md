# Interview project for Senior Data Scientist Position

Small Sales Forecasting Project for a Senior Data Scientist position application.

## Train & save model
Run the code using the following steps:

1. run: make install
2. update dataset_path in config.yaml to point to your dataset
3. run: python train_model.py (default config) OR python train_model.py -c your_config_path.yaml

Default model assumes a 7 day forecast horizon.

## Data
Data not included: private to the company.
Minimal data exploration can be found in: [notebooks/quick_EDA.ipynb](https://github.com/miglash/sds_interview_project/blob/main/notebooks/quick_EDA.ipynb)

The waste to sales ratio is on average ~20%, so a forecast model needs to have a lower error to be useful.

## Preprocessing
1. Sales data aggregated
2. Holidays "detected as outliers" and imputed as median value of the same weekday
3. No missing data --> no further imputation
4. Added mean and standard deviation of training sample as further features
5. Input time-length left as a free parameter: surprisingly short 10-14 days lead to best performance for model tested.

## Results
Results from 3 models can be compared in [notebooks/results.ipynb](https://github.com/miglash/sds_interview_project/blob/main/notebooks/evaluation.ipynb)

1. Naive baseline - predict mean of inputs - 220% MAPE (Mean Absolute Percentage Error). This is a bit misleading since weekend/weekday sales vary dramatically.
2. Linear baseline - 16% MAPE
3. Xgboost model - 9% MAPE

Interestingly, performance degrades with time-folds for xgboost (12% MAPE in last fold), but improves for linear model (13% MAPE in last fold)

## Further improvements
1. Dataset logging
2. Model logging, e.g. MLFlow
3. Extend training for Random/Grid search over parameters
4. Code clean-up: validating params, paths, config etc + Documentation
5. Dataset (& any other defaults) generated from config
6. Countless model improvements
7. ...


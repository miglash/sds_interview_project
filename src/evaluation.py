from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
from typing import Dict, List

def evaluate(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict:
    """ Given a list of prediction calculate metrics"""
    results = {"RMSE": rmse(Y_true, Y_pred),
               "MAPE": mape(Y_true, Y_pred)}
    return results

def evaluate_cross_val(results: List) -> Dict:
    """ Given a nested list of predictions calculate metrics"""
    cv_results = {}

    all_true, all_pred = [],[]
    for i_val, cv_val in enumerate(results):
        Y_true, Y_pred = cv_val
        cv_results[f"CV FOLD {i_val}"] = evaluate(Y_true, Y_pred)
        
        #Un-nest
        all_true.append(Y_true)
        all_pred.append(Y_pred)

    all_true = np.concat(all_true, axis=0)
    all_pred = np.concat(all_pred, axis=0)
    
    cv_results[f"AVG"] = evaluate(all_true, all_pred)
    return cv_results
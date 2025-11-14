from src.model import BaselineModel
import numpy as np

#An example test
def test_last_baseline():
    model = BaselineModel(method="last")

    X = np.random.rand(20, 30)
    Y = np.zeros([20, 1])

    model.fit(X,Y)
    Y_pred = model.predict(X)

    assert (Y_pred[:,-1] == X[:,-1]).all(), "Incorrect last value prediction"
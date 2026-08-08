import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is( n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        return np.round(X @ weights,5)
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Round to 5 decimal places
        # Compute mean squared error
        ans=np.sum((model_prediction-ground_truth)**2)/len(model_prediction)
        return np.round(ans,5)
        pass

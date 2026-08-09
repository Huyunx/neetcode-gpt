import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        dot = x @ w
        res=dot+b


        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        if(activation=="sigmoid"):
            ans=1.0/(1.0+np.exp(-res))
            return np.round(ans,5)
        else:
            return np.round(max(0.0,res),5)

        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        pass

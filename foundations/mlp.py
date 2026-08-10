import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        def relu(x):
            return np.maximum(0,x)
        a=x
        z=x
        for i in range(0,len(weights)):
            w=weights[i]
            w=np.swapaxes(w,0,1)
            b=biases[i]
            z= w @ a +b
            a=relu(z)
        
        return z
        pass

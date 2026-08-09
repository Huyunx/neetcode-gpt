import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        def Linear(W,b,x):
            return (W @ x +b)
        def relu(x):
            return np.maximum(0,x)
        z1=Linear(W1,b1,x)
        midlayer=relu(z1)
        z2=Linear(W2,b2,midlayer)
        predictions=relu(z2)
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        n=len(y_true)
        L=np.mean((predictions-y_true)**2)
        dldp=2*(predictions-y_true)/n
        dpdz=(predictions>0).astype(float)

        dldmidlayer=(dldp*dpdz) @ W2
        dmidlayerdz1=(midlayer>0).astype(float)

        di={
            "loss":np.round( L,4),
            "dW2": np.round(np.outer(dldp*dpdz,midlayer),4),
            "db2": np.round(dldp*dpdz,4),
            "dW1": np.round(np.outer(dldmidlayer*dmidlayerdz1,x),4),
            "db1": np.round(dldmidlayer*dmidlayerdz1,4)
        }
       
        return di

        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        pass

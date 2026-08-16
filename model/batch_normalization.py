import numpy as np
import torch
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        m=momentum
        x=torch.Tensor(x)
        gamma=torch.Tensor(gamma)
        beta=torch.Tensor(beta)
        running_mean = torch.Tensor(running_mean)
        running_var = torch.Tensor(running_var)
        if(training):
            # During training: normalize using batch statistics, then update running stats
            muB=torch.mean(x,dim=0)
            variance=torch.var(x,dim=0,unbiased=False)

            x_hat=torch.Tensor()
          
            x_hat=(x-muB)/torch.sqrt(variance+eps)
            y=gamma*x_hat+beta
     
            
            running_mean=(1-m)*running_mean+m*muB
            running_var=(1-m)*running_var+m*variance
            
        else:
            x_hat=(x-running_mean)/torch.sqrt(running_var+eps)
            y=gamma*x_hat+beta
        y = [[round(v, 4) for v in row] for row in y.tolist()]
        running_mean = [round(v, 4) for v in running_mean.tolist()]
        running_var = [round(v, 4) for v in running_var.tolist()]

        return y, running_mean, running_var
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        pass

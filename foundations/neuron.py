import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        p = np.dot(x,w)+ b
        if activation == "relu":
            z = max(0.0,p)
        elif activation == "sigmoid":
            z = 1.0 / (1.0 + np.exp(-p))
        else:
            z = p

        return round(float(z),5)

        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        pass

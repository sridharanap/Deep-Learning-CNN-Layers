import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.loss = None
        self.eps_yhat = None

    def forward(self, prediction_tensor, label_tensor):
        eps_yhat = prediction_tensor + np.finfo(float).eps
        self.eps_yhat = eps_yhat
        ln_matrix = -np.log(eps_yhat)
        ce_matrix = np.sum(ln_matrix * label_tensor, axis=1)
        loss = np.sum(ce_matrix)
        self.loss = loss
        return loss

    def backward(self, label_tensor):
        error_tensor = -label_tensor / self.eps_yhat
        return error_tensor

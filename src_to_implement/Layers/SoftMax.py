from .Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        row_max = np.max(input_tensor, axis=1, keepdims=True)
        input_shift = input_tensor - row_max
        exp_input = np.exp(input_shift)
        exp_sums = np.sum(exp_input, axis=1, keepdims=True)
        output_tensor = exp_input / exp_sums
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        elem_mul = error_tensor * self.output_tensor
        sum_matrix = np.sum(elem_mul, axis=1, keepdims=True)
        gradient_tensor = self.output_tensor * (error_tensor - sum_matrix)
        return gradient_tensor

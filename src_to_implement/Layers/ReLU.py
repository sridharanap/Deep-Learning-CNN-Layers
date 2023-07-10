from .Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = np.maximum(0, input_tensor)
        return output_tensor

    def backward(self, error_tensor):
        gradient_tensor = error_tensor*np.where(self.input_tensor <= 0, 0, 1)
        return gradient_tensor

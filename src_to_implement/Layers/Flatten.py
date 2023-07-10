from .Base import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        self.input_shape = input_tensor.shape
        flat_output = np.ravel(input_tensor).reshape(batch_size, -1)
        return flat_output

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)

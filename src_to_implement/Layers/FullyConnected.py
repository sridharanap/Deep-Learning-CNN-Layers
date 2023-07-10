from .Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0,1,(input_size+1,output_size))
        self._optimizer = None
        self.input_tensor = None
        self.gradient_tensor = None
        self._gradient_weights = None
        self._weights_initializer=None
        self._bias_initializer=None
        # self.bias=np.random.uniform(0,1,output_size)

    def initialize(self,weights_initializer,bias_initializer):
        self.weights=weights_initializer.initialize(self.weights.shape,self.input_size,self.output_size)
        self.weights[-1,:]=bias_initializer.initialize(self.output_size,self.input_size,self.output_size)
        # self.weights=np.vstack(weights,bias)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        input_bias_tensor = np.hstack((input_tensor, np.ones((batch_size, 1))))
        output_tensor = np.dot(input_bias_tensor, self.weights)
        self.input_tensor = input_bias_tensor
        return output_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer


    def backward(self, error_tensor):
        weights_without_bias=self.weights[:-1,:]
        self.gradient_tensor = np.dot(error_tensor, weights_without_bias.T)
        # self.gradient_tensor = self.gradient_tensor[:, :-1]
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return self.gradient_tensor
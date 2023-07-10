from .Base import BaseLayer
import numpy as np
import copy
from scipy import signal


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.rand(self.num_kernels, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self.input_tensor = None

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        # to try directly passing weights shape without constructor initialization
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # if len(self.stride_shape)==2:
        #     stride_x, stride_y = self.stride_shape
        # else:
        #     stride_x=self.stride_shape
        batch_size = input_tensor.shape[0]
        input_channel_size = input_tensor.shape[1]
        # if len(self.convolution_shape) == 3:
        #     output_channel_size, kernelspat_x, kernelspat_y = self.convolution_shape
        # elif len(self.convolution_shape) == 2:
        #     output_channel_size, kernelspat_y = self.convolution_shape
        #     kernelspat_x = 1
        output = np.zeros((batch_size, self.num_kernels, *input_tensor.shape[2:]))
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(input_channel_size):
                    output[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], "same")
                output[b, k] += self.bias[k]
        if len(self.convolution_shape) == 3:  # filters the output matrix based on strides taken
            output = output[:, :, 0::self.stride_shape[0], 0::self.stride_shape[1]]
        else:
            output = output[:, :, 0::self.stride_shape[0]]
        return output

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        # two copies of optimizer object each for weights & bias gradient calculation
        optimizer_1 = copy.deepcopy(optimizer)
        optimizer_2 = copy.deepcopy(optimizer)
        self._optimizer = [optimizer_1, optimizer_2]

    def backward(self, error_tensor):
        self._gradient_weights = np.zeros(self.weights.shape)
        batch_size=self.input_tensor.shape[0]
        if len(self.convolution_shape) == 3:
            axis = (0, 2, 3)
        else:
            axis = (0, 2)
        self._gradient_bias = np.sum(error_tensor, axis=axis)
        output_error_tensor = np.zeros(self.input_tensor.shape)
        input_channel_size = self.input_tensor.shape[1]
        # padding
        filter_matrix = np.array(self.convolution_shape[1:])
        initial_padding = np.floor(filter_matrix / 2).astype(int)
        final_padding = filter_matrix - initial_padding - 1
        #condition for 3D/2D
        if len(filter_matrix) == 2:
            padding_width = [(0, 0), (0, 0), (initial_padding[0], final_padding[0]),(initial_padding[1], final_padding[1])]
        else:
            padding_width = [(0, 0), (0, 0), (initial_padding[0], final_padding[0])]
        padded_input = np.pad(self.input_tensor, pad_width=padding_width, constant_values=0)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(input_channel_size):
                    # upsampling error_tensor according to stride
                    output_shape=self.input_tensor.shape[2:]
                    upsampling_error = np.zeros(output_shape)
                    if len(output_shape) == 2:  # In-case of 2D convolution
                        upsampling_error[0::self.stride_shape[0], 0::self.stride_shape[1]] = error_tensor[b, k]
                    else:
                        upsampling_error[0::self.stride_shape[0]] = error_tensor[b, k]
                    output_weight = signal.correlate(padded_input[b, c], upsampling_error, "valid")
                    self._gradient_weights[k, c] += output_weight
                    output_error = signal.convolve(upsampling_error, self.weights[k, c],"same")  # Upsampled error being padded to the output error_tensor
                    output_error_tensor[b,c] += output_error
        if self._optimizer is not None:
            self.weights = self._optimizer[0].calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer[1].calculate_update(self.bias, self._gradient_bias)
        return output_error_tensor

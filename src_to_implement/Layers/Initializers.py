import numpy as np


class Constant:
    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.full(weights_shape, self.constant)
        return weights


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.uniform(0, 1, weights_shape)
        return weights


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return weights


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        weights = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return weights

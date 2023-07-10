import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - (self.learning_rate * gradient_tensor)
        return updated_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate=0.9):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        # if self.velocity is None:
        #     self.velocity = np.zeros_like(gradient_tensor)
        # else:
        self.velocity = (self.momentum_rate * self.velocity) - (self.learning_rate * gradient_tensor)
        updated_weights = weight_tensor + self.velocity
        return updated_weights


class Adam:
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.velocity = 0
        self.r_moment = 0
        self.count = 1
        self.epsilon = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):
        # if self.velocity is None:
        #     self.velocity=np.zeros_like(gradient_tensor)
        #     self.r_moment=np.zeros_like(gradient_tensor)
        # else:
        self.velocity = (self.mu * self.velocity) + ((1 - self.mu) * gradient_tensor)
        self.r_moment = (self.rho * self.r_moment) + ((1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor))
        v_hat = self.velocity / (1 - self.mu ** self.count)
        r_hat = self.r_moment / (1 - self.rho ** self.count)
        self.count += 1
        updated_weights = weight_tensor - (self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps)))
        return updated_weights

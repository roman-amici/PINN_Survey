from PINN_Base.base_v1 import PINN_Base
import tensorflow as tf
import numpy as np


class Random_Fourier(PINN_Base):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers,
                 fourier_stddev=20,
                 **kwargs):

        self.fourier_stddev = fourier_stddev

        super().__init__(lower_bound, upper_bound, layers, **kwargs)

    def _init_params(self):

        shape_0 = [self.layers[0], self.layers[1]//2]

        self.fourier_weights = tf.constant(np.random.normal(
            scale=self.fourier_stddev, size=shape_0),  dtype=self.dtype)

        # don't generate params for the first layer
        layers = self.layers[1:]

        self.weights, self.biases = self._init_NN(layers)

    def _NN(self, X, weights, biases):
        activations = []

        H = 2.0 * (X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0
        activations.append(H)

        H_1 = tf.sin(tf.matmul(H, self.fourier_weights))
        H_2 = tf.cos(tf.matmul(H, self.fourier_weights))

        H = tf.concat([H_1, H_2], axis=1)
        activations.append(H)

        for l in range(len(weights)-1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            activations.append(H)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)

        return Y, activations

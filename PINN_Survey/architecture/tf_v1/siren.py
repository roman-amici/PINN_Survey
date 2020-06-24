from PINN_Base.base_v1 import PINN_Base
import tensorflow as tf
import numpy as np


class Siren(PINN_Base):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers,
                 init_freqs=30,
                 **kwargs):

        self.init_freqs = init_freqs

        super().__init__(lower_bound, upper_bound, layers, **kwargs)

    def _NN(self, X, weights, biases):

        activations = []

        H = 2.0 * (X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0
        activations.append(H)

        for l in range(len(weights)-1):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
            activations.append(H)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)

        return Y, activations

    def _siren_init(self, size):

        in_dim = size[0]

        lb = -np.sqrt(6/in_dim)
        ub = np.sqrt(6/in_dim)

        return tf.Variable(
            tf.random.uniform(
                shape=size, min_val=lb, max_val=ub, dtype=self.dtype),
            dtype=self.dtype)

    def _init_NN(self, layers):
        weights = []
        biases = []

        shape_0 = [layers[0], layers[1]]
        bound_0 = np.sqrt(6/layers[0])

        # Spread the initial encoding over a wider set of frequencies
        W_0 = tf.Variable(self.init_freqs *
                          tf.random.uniform(shape_0, -bound_0, bound_0), dtype=self.dtype)
        b_0 = b = tf.Variable(
            tf.zeros([1, layers[1]], dtype=self.dtype), dtype=self.dtype)
        weights.append(W_0)
        biases.append(b_0)

        for l in range(1, len(layers)-1):
            W = self._xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(
                tf.zeros([1, layers[l+1]], dtype=self.dtype), dtype=self.dtype)
            weights.append(W)
            biases.append(b)

        return weights, biases

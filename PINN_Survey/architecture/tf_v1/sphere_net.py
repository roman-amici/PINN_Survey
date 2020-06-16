from PINN_Base.base_v1 import PINN_Base
import tensorflow as tf


class Sphere_Net(PINN_Base):

    def _NN(self, X, weights, biases):

        activations = []

        H = 2.0 * (X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0
        activations.append(H)

        for l in range(len(weights)-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            activations.append(H)

        W = weights[-2]
        biases = biases[-2]
        Z = tf.square(tf.add(tf.matmul(H, W), b))
        H = Z / tf.reduce_sum(Z, axis=1)[:, None]
        activations.append(H)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)

        return Y, activations

    def get_architecture_description(self):
        params = self._count_params()
        return {
            "arch_name": "sphere_net",
            "n_params": params,
            "shape": self.layers[:],
            "dtype": "float32" if self.dtype == tf.float32 else "float64"
        }

from PINN_Base.base_v1 import PINN_Base
import tensorflow as tf


class PINN_Resnet(PINN_Base):

    def _NN(self, X, weights, biases):

        activations = []

        Z = 2.0 * (X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0
        activations.append(Z)

        W = weights[0]
        b = biases[0]
        Z = tf.tanh(tf.add(tf.matmul(Z, W), b))
        activations.append(Z)

        F = Z

        for l in range(1, len(weights) - 1):
            W = weights[l]
            b = biases[l]
            Z = tf.tanh(tf.add(tf.matmul(Z, W), b))
            activations.append(Z)

            if l % 2 == 1:
                F = F + Z
                Z = F

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(Z, W), b)

        return Y, activations

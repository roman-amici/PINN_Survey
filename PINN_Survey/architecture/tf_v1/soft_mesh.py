
from PINN_Base.base_v1 import PINN_Base
import tensorflow as tf


class Soft_Mesh(PINN_Base):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers_approx,
                 layers_mesh,
                 **kwargs
                 ):
        assert(layers_approx[-2] == layers_mesh[-1])

        self.output_dim = layers_approx[-1]

        # Output just the weighted basis functions which make up the final layer
        layers = layers_approx[:-1]

        self.layers_mesh = layers_mesh

        super().__init__(lower_bound, upper_bound, layers, **kwargs)

    def _init_params(self):

        self.weights_mesh, self.biases_mesh = self._init_NN(self.layers_mesh)

        super()._init_params()

    def _forward_mesh(self, X, weights, biases):

        scores, activations = self._NN(X, weights, biases)

        return tf.nn.softmax(scores), activations

    def _forward(self, X):

        probs, acitvations_probs = self._forward_mesh(
            X, self.weights_mesh, self.biases_mesh)

        basis_functions, activations_basis = self._NN(
            X, self.weights, self.biases)

        if X == self.X:
            self.probs = probs
            self.activations_probs = acitvations_probs
            self.basis_functions = basis_functions
            self.activations_basis = activations_basis

        # Take the combination of basis functions multiplied by probabilities.
        return tf.reduce_sum(basis_functions * probs, axis=1)[:, None]

    def get_output_dim(self):
        return self.output_dim

    def get_probs(self, X):
        return self.sess.run(self.probs, {self.X: X})

    def get_basis_functions(self, X):
        return self.sess.run(self.basis_functions, {self.X: X})

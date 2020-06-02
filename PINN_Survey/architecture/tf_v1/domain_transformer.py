import PINN_Base.base_v1 as base_v1
import tensorflow as tf

'''
This is an implementation of the (unnamed) 
"Improved fully-connected neural architecture" from 
UNDERSTANDING AND MITIGATING GRADIENT PATHOLOGIES IN
PHYSICS-INFORMED NEURAL NETWORKS (Wang, 2020).

I have taken the liberty of naming it based on the authors
likening it to the Transformer. Unlike the transformer, it doesn't
transform the outputs, Z, but rather the inputs, X.
'''


class Domain_Transformer(base_v1.PINN_Base):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 input_dim,
                 output_dim,
                 width,
                 depth,
                 **kwargs):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.width = width
        self.depth = depth

        layers = [self.input_dim] + [self.width] * depth + [self.output_dim]

        super().__init__(lower_bound, upper_bound, layers, **kwargs)

    def _init_params(self):

        # Two layer neural network
        transformer_shape = [self.input_dim, self.width, self.width]

        # The first encoder network
        self.weights_T1, self.biases_T1 = self._init_NN(transformer_shape)
        # The second encoder network
        self.weights_T2, self.biases_T2 = self._init_NN(transformer_shape)

        # The normal "forward" network is initialized by parent
        super()._init_params()

    def _domain_transformer_forward(self, X, T1, T2, weights, biases):
        activations = []

        H = 2.0 * (X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0
        activations.append(H)

        for l in range(len(weights)-1):
            W = weights[l]
            b = biases[l]
            Z = tf.tanh(tf.add(tf.matmul(H, W), b))
            H = (1 - Z) * T1 + Z * T2
            activations.append(H)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)

        return Y, activations

    def _forward(self, X):

        T1, activations_T1 = self._NN(X, self.weights_T1, self.biases_T2)
        T2, activations_T2 = self._NN(X, self.weights_T2, self.biases_T2)

        U, activations = self._domain_transformer_forward(
            X, T1, T2, self.weights, self.biases)

        if X == self.X:
            self.T1 = T1
            self.T2 = T2
            self.activations = activations
            self.Activations_T1 = activations_T1
            self.Activations_T2 = activations_T2

        return U

    def get_T1(self, X):
        return self.sess.run(self.T1, {self.X: X})

    def get_T2(self, X):
        return self.sess.run(self.T2, {self.X: X})

    def _count_params(self):
        params_main = super()._count_params()

        params_T1_weights = self._size_of_variable_list(self.weights_T1)
        params_T1_biases = self._size_of_variable_list(self.biases_T1)

        params_T2_weights = self._size_of_variable_list(self.weights_T2)
        params_T2_biases = self._size_of_variable_list(self.biases_T2)

        return params_main + params_T1_weights + params_T1_biases + params_T2_weights + params_T2_biases

    def get_architecture_description(self):
        params = self._count_params()
        return {
            "arch_name": "domain_transformer",
            "n_params": params,
            "shape_main": self.layers[:],
            "shape_T1": [self.input_dim, self.width, self.output_dim],
            "shape_T2": [self.input_dim, self.width, self.output_dim],
            "dtype": "float32" if self.dtype == tf.float32 else "float64"
        }

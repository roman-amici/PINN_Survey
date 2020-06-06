import tensorflow as tf
from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh


class Sphere_Mesh(Soft_Mesh):

    def _forward_mesh(self, X, weights, biases):
        scores, activations = self._NN(X, weights, biases)

        # Normalize using the square of the scores rather than the softmax
        z = tf.square(scores)
        norm = tf.reduce_sum(z, axis=1)[:, None]

        return z/norm, activations

    def get_architecture_description(self):
        params = self._count_params()
        return {
            "arch_name": "sphere_mesh",
            "n_params": params,
            "shape_fit": self.layers[:],
            "shape_mesh": self.layers_mesh[:],
            "dtype": "float32" if self.dtype == tf.float32 else "float64"
        }

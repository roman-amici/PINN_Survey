from PINN_Base.base_v1 import PINN_Base
from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
import tensorflow as tf
import numpy as np


def Poisson1DResidual(base_class):

    class WithPoisson1DResidual(base_class):
        def f(self, X):
            pi = np.pi

            t1 = -0.1 * (8 * pi)**2 * tf.sin(8 * pi * X[:, 0])
            t2 = 2 * (80**2) * tf.tanh(80 *
                                       X[:, 0]) * (1 / tf.cosh(80 * X[:, 0]))**2

            return t1 - t2

        def _residual(self, u, x, _=None):
            u_x = tf.gradients(u, x)[0]

            u_xx = tf.gradients(u_x[:, 0], x)[0]

            return u_xx[:, 0] - self.f(x)

    return WithPoisson1DResidual


@Poisson1DResidual
class Poisson(PINN_Base):
    pass


@Poisson1DResidual
class Poisson_Soft_Mesh(Soft_Mesh):
    pass


@Poisson1DResidual
class Poisson_Domain_Transformer(Domain_Transformer):
    pass

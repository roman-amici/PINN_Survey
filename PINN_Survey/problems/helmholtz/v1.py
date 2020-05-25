from PINN_Base.base_v1 import PINN_Base
from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
import tensorflow as tf
import numpy as np


def q(self, X):
    a = self.a
    b = self.b
    pi = np.pi
    sin = tf.sin

    # We implicitly remove the k^2 sin(x)*sin(y) term since it cancels
    # Otherwise we would essentially be leaking the solution in the differential equation
    return -(a*pi)**2 * sin(a*pi*X[:, 0]) * sin(b*pi*X[:, 1]) - \
        (b*pi)**2 * sin(a*pi*X[:, 0])*sin(b*pi*X[:, 1])


def residual(self, u, x):
    du = tf.gradients(u, x)[0]

    U_2x = tf.gradients(du[:, 0], x)[0]
    U_2y = tf.gradients(du[:, 1], x)[0]

    u_xx = U_2x[:, 0]
    u_yy = U_2y[:, 1]

    return u_xx + u_yy - q(self, x)


class Helmholtz(PINN_Base):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers,
                 a, b,
                 **kwargs):

        self.a = a
        self.b = b
        super().__init__(lower_bound, upper_bound, layers, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Helmholtz_Soft_Mesh(Soft_Mesh):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers_approx,
                 layers_mesh,
                 a, b,
                 **kwargs):

        self.a = a
        self.b = b

        super().__init__(lower_bound, upper_bound, layers_approx, layers_mesh, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Helmholtz_Domain_Transformer(Domain_Transformer):
    def __init__(self,
                 lower_bound,
                 upper_bound,
                 width,
                 depth,
                 a,
                 b,
                 **kwargs):

        self.a = a
        self.b = b
        super().__init__(lower_bound, upper_bound, 2, 1, width, depth, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)

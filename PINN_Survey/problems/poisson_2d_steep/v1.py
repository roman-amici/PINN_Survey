from PINN_Base.base_v1 import PINN_Base
from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.sphere_mesh import Sphere_Mesh
from PINN_Survey.architecture.tf_v1.sphere_net import Sphere_Net
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
import tensorflow as tf
import numpy as np


def Poisson2DResidual(base_class):

    class WithPoisson2DResidual(base_class):
        def f(self, X):
            pi = np.pi
            sin = tf.sin
            tanh = tf.tanh
            cosh = tf.cosh

            t1 = -0.1 * (2 * pi)**2 * sin(2 * pi * X[:, 0]) - 2 * \
                100 * tanh(10 * X[:, 0]) * (1 / cosh(10 * X[:, 0]))**2
            t2 = 0.1 * sin(2 * pi * X[:, 0]) + tanh(2 * pi * X[:, 0])
            t3 = -(2 * pi)**2 * sin(2 * pi * X[:, 1])

            return t1 * sin(2 * pi * X[:, 1]) + t2 * t3

        def _residual(self, u, x, _=None):
            u_1 = tf.gradients(u, x)[0]

            u_x = u_1[:, 0]
            u_y = u_1[:, 1]

            u_xx = tf.gradients(u_x, x)[0][:, 0]
            u_yy = tf.gradients(u_y, x)[0][:, 1]

            return u_xx + u_yy - self.f(x)

    return WithPoisson2DResidual


@Poisson2DResidual
class Poisson(PINN_Base):
    pass


@Poisson2DResidual
class Poisson_Soft_Mesh(Soft_Mesh):
    pass


@Poisson2DResidual
class Poisson_Domain_Transformer(Domain_Transformer):
    pass


@Poisson2DResidual
class Poisson_Sphere_Mesh(Sphere_Mesh):
    pass


@Poisson2DResidual
class Poisson_Sphere_Net(Sphere_Net):
    pass

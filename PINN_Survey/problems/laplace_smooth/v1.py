from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.sphere_mesh import Sphere_Mesh
import tensorflow as tf
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
from PINN_Base.base_v1 import PINN_Base


def residual(self, u, x):

    du1 = tf.gradients(u, x)[0]

    u_x = du1[:, 0]
    u_y = du1[:, 1]

    u_xx = tf.gradients(u_x, x)[0][:, 0]
    u_yy = tf.gradients(u_y, x)[0][:, 1]

    return u_xx + u_yy


class Laplace(PINN_Base):

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Laplace_Soft_Mesh(Soft_Mesh):

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Laplace_Domain_Transformer(Domain_Transformer):

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Laplace_Sphere_Mesh(Sphere_Mesh):

    def _residual(self, u, x, _=None):
        return residual(self, u, x)

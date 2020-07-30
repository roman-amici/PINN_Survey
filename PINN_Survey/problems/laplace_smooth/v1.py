from PINN_Base.base_v1 import PINN_Base
from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.sphere_mesh import Sphere_Mesh
from PINN_Survey.architecture.tf_v1.sphere_net import Sphere_Net
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
import tensorflow as tf


def LaplaceResidual(base_class):

    class WithLaplaceResidual(base_class):

        def _residual(self, u, x, _=None):

            du1 = tf.gradients(u, x)[0]

            u_x = du1[:, 0]
            u_y = du1[:, 1]

            u_xx = tf.gradients(u_x, x)[0][:, 0]
            u_yy = tf.gradients(u_y, x)[0][:, 1]

            return u_xx + u_yy

    return WithLaplaceResidual


@LaplaceResidual
class Laplace(PINN_Base):
    pass


@LaplaceResidual
class Laplace_Soft_Mesh(Soft_Mesh):
    pass


@LaplaceResidual
class Laplace_Domain_Transformer(Domain_Transformer):
    pass


@LaplaceResidual
class Laplace_Sphere_Mesh(Sphere_Mesh):
    pass


@LaplaceResidual
class Laplace_Sphere_Net(Sphere_Net):
    pass

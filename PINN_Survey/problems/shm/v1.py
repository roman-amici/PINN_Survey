import tensorflow as tf
from PINN_Base.base_v1 import PINN_Base
from PINN_Survey.architecture.tf_v1.sphere_net import Sphere_Net
from PINN_Survey.architecture.tf_v1.sphere_mesh import Sphere_Mesh


def MakeSHM(base_class):

    class WithSHM(base_class):

        def __init__(self, omega, *args, **kwargs):

            self.omega = omega

            super().__init__(*args, **kwargs)

        def _residual(self, U, X, U_true=None):

            U_x = tf.gradients(U, X)[0][:, 0]
            U_xx = tf.gradients(U_x, X)[0]

            if U_true is not None:
                return U_xx[:, 0] + (self.omega**2) * U_true[:, 0]
            else:
                return U_xx[:, 0] + (self.omega**2) * U[:, 0]

    return WithSHM


@MakeSHM
class SHM(PINN_Base):
    pass


@MakeSHM
class SHMSphereNet(Sphere_Net):
    pass


@MakeSHM
class SHMSphereMesh(Sphere_Mesh):
    pass

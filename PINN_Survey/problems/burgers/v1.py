from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.sphere_mesh import Sphere_Mesh
from PINN_Survey.architecture.tf_v1.sphere_net import Sphere_Net
from PINN_Survey.architecture.tf_v1.siren import Siren
from PINN_Survey.architecture.tf_v1.random_fourier import Random_Fourier
from PINN_Survey.architecture.tf_v1.resnet import PINN_Resnet
import tensorflow as tf
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
from PINN_Base.base_v1 import PINN_Base


def BurgersResidual(BaseClass):

    class BurgersWithResidual(BaseClass):

        def __init__(self, nu, *args, **kwargs):
            self.nu = nu

            super().__init__(*args, **kwargs)

        def _residual(self, u, x, _=None):
            du = tf.gradients(u, x)[0]

            u_x = du[:, 0]
            u_t = du[:, 1]

            du2 = tf.gradients(u_x, x)[0]
            u_xx = du2[:, 0]

            return u_t + u[:, 0] * u_x - self.nu * u_xx

    return BurgersWithResidual


@BurgersResidual
class Burgers(PINN_Base):
    pass


@BurgersResidual
class Burgers_Soft_Mesh(Soft_Mesh):
    pass


@BurgersResidual
class Burgers_Domain_Transformer(Domain_Transformer):
    pass


@BurgersResidual
class Burgers_Sphere_Mesh(Sphere_Mesh):
    pass


@BurgersResidual
class Burgers_Sphere_Net(Sphere_Net):
    pass


@BurgersResidual
class Burgers_Siren(PINN_Base):
    pass


@BurgersResidual
class Burgers_Random_Fourier(Random_Fourier):
    pass


@BurgersResidual
class Burgers_Resnet(PINN_Resnet):
    pass

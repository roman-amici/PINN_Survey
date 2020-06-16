from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.sphere_mesh import Sphere_Mesh
from PINN_Survey.architecture.tf_v1.sphere_net import Sphere_Net
import tensorflow as tf
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
from PINN_Base.base_v1 import PINN_Base


def residual(self, u, x):
    du = tf.gradients(u, x)[0]

    u_x = du[:, 0]
    u_t = du[:, 1]

    du2 = tf.gradients(u_x, x)[0]
    u_xx = du2[:, 0]

    return u_t + u[:, 0]*u_x - self.nu*u_xx


class Burgers(PINN_Base):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers,
                 nu,
                 **kwargs):

        self.nu = nu
        super().__init__(lower_bound, upper_bound, layers, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Burgers_Soft_Mesh(Soft_Mesh):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers_approx,
                 layers_mesh,
                 nu,
                 **kwargs
                 ):

        self.nu = nu
        super().__init__(lower_bound, upper_bound, layers_approx, layers_mesh, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Burgers_Domain_Transformer(Domain_Transformer):
    def __init__(self,
                 lower_bound,
                 upper_bound,
                 width,
                 depth,
                 nu,
                 **kwargs
                 ):

        self.nu = nu
        super().__init__(lower_bound, upper_bound, 2, 1, width, depth, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Burgers_Sphere_Mesh(Sphere_Mesh):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers_approx,
                 layers_mesh,
                 nu,
                 **kwargs
                 ):

        self.nu = nu
        super().__init__(lower_bound, upper_bound, layers_approx, layers_mesh, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Burgers_Sphere_Net(Sphere_Net):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers,
                 nu,
                 **kwargs):

        self.nu = nu
        super().__init__(lower_bound, upper_bound, layers, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)

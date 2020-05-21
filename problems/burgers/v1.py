from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
import tensorflow as tf
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer


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

    def _residual(self, u, x, _):
        du = tf.gradients(u, x)[0]

        u_x = du[:, 0]
        u_t = du[:, 1]

        du2 = tf.gradients(u_x, x)[0]
        u_xx = du2[:, 0]

        return u_t + u[:, 0]*u_x - self.nu*u_xx


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

    def _residual(self, u, x, _):
        du = tf.gradients(u, x)[0]

        u_x = du[:, 0]
        u_t = du[:, 1]

        du2 = tf.gradients(u_x, x)[0]
        u_xx = du2[:, 0]

        return u_t + u[:, 0]*u_x - self.nu*u_xx

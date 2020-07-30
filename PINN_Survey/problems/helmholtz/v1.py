from PINN_Base.base_v1 import PINN_Base
from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.sphere_mesh import Sphere_Mesh
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
from PINN_Survey.architecture.tf_v1.sphere_net import Sphere_Net
from PINN_Survey.architecture.tf_v1.siren import Siren
from PINN_Survey.architecture.tf_v1.random_fourier import Random_Fourier
import tensorflow as tf
import numpy as np


def HelmholtzResidual(base_class):

    class HelmholtzWithResidual(base_class):
        '''
        This implements the residual for the particular solution 
        sin(ax) * sin(by).
        Consider making this an optional input parameter
        for more general solutions to the helmholtz equation.
        '''

        def __init__(self, a, b, *args, **kwargs):
            self.a = a
            self.b = b
            super().__init__(*args, **kwargs)

        def q(self, X):
            a = self.a
            b = self.b
            pi = np.pi
            sin = tf.sin

            return -(a * pi)**2 * sin(a * pi * X[:, 0]) * sin(b * pi * X[:, 1]) - \
                (b * pi)**2 * sin(a * pi * X[:, 0]) * sin(b * pi * X[:, 1]
                                                          ) + sin(a * pi * X[:, 0]) * sin(b * pi * X[:, 1])

        def _residual(self, u, x, _=None):
            du = tf.gradients(u, x)[0]

            U_2x = tf.gradients(du[:, 0], x)[0]
            U_2y = tf.gradients(du[:, 1], x)[0]

            u_xx = U_2x[:, 0]
            u_yy = U_2y[:, 1]

            return u_xx + u_yy + u[:, 0] - self.q(x)

    return HelmholtzWithResidual


@HelmholtzResidual
class Helmholtz(PINN_Base):
    pass


@HelmholtzResidual
class Helmholtz_Soft_Mesh(Soft_Mesh):
    pass


@HelmholtzResidual
class Helmholtz_Domain_Transformer(Domain_Transformer):
    pass


@HelmholtzResidual
class Helmholtz_Sphere_Mesh(Sphere_Mesh):
    pass


@HelmholtzResidual
class Helmholtz_Sphere_Net(Sphere_Net):
    pass


@HelmholtzResidual
class Helmholtz_Siren(Siren):
    pass


@HelmholtzResidual
class Helmholtz_Random_Fourier(Random_Fourier):
    pass

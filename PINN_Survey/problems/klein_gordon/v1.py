from PINN_Base.base_v1 import PINN_Base
from PINN_Survey.architecture.tf_v1.soft_mesh import Soft_Mesh
from PINN_Survey.architecture.tf_v1.sphere_mesh import Sphere_Mesh
from PINN_Survey.architecture.tf_v1.domain_transformer import Domain_Transformer
import tensorflow as tf
import numpy as np


def f(self, X):
    # Forcing function corresponding to the chosen analytic solution
    pi = np.pi
    x = X[:, 0]
    t = X[:, 1]

    t1 = -x*25*pi**2 * tf.cos(5*pi*t) + 6*t*x**3
    t2 = 6*self.alpha*x*(t**3)
    t3 = self.beta*(x*tf.cos(5*pi*t) + (x*t)**3)
    t4 = self.gamma*(x*tf.cos(5*pi*t) + (x*t)**3)**self.k

    return t1 + t2 + t3 + t4


def residual(self, u, x):
    du = tf.gradients(u, x)[0]

    U_2x = tf.gradients(du[:, 0], x)[0]
    U_2t = tf.gradients(du[:, 1], x)[0]

    u_xx = U_2x[:, 0]
    u_tt = U_2t[:, 1]

    lhs = u_tt + self.alpha*u_xx + self.beta * \
        u[:, 0] + self.gamma * (u[:, 0])**self.k

    return lhs - f(self, x)


class Klein_Gordon(PINN_Base):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers,
                 alpha, beta, gamma, k,
                 ** kwargs):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        super().__init__(lower_bound, upper_bound, layers, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Klein_Gordon_Soft_Mesh(Soft_Mesh):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers_approx,
                 layers_mesh,
                 alpha, beta, gamma, k,
                 **kwargs):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k

        super().__init__(lower_bound, upper_bound, layers_approx, layers_mesh, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Klein_Gordon_Domain_Transformer(Domain_Transformer):
    def __init__(self,
                 lower_bound,
                 upper_bound,
                 width,
                 depth,
                 alpha, beta, gamma, k,
                 **kwargs):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        super().__init__(lower_bound, upper_bound, 2, 1, width, depth, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)


class Klein_Gordon_Sphere_Mesh(Sphere_Mesh):

    def __init__(self,
                 lower_bound,
                 upper_bound,
                 layers_approx,
                 layers_mesh,
                 alpha, beta, gamma, k,
                 **kwargs):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k

        super().__init__(lower_bound, upper_bound, layers_approx, layers_mesh, **kwargs)

    def _residual(self, u, x, _=None):
        return residual(self, u, x)

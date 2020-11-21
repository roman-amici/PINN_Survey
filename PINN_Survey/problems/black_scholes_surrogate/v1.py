import tensorflow as tf
from PINN_Base.base_v1 import PINN_Base


def BlackScholesResidual(BaseClass):

    class BlackScholesWithResidual(BaseClass):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def _residual(self, u, x, _=None):
            s = x[:, 0]
            r = x[:, 2]
            sigma = x[:, 3]

            du_1 = tf.gradients(u, x)[0]
            u_s = du_1[:, 0]
            u_t = du_1[:, 1]
            u_ss = tf.gradients(u_s, x)[0][:, 0]

            # Compute "the greeks"
            if x == self.X:
                self.delta = u_s
                self.gamma = u_ss
                self.theta = u_t
                self.rho = du_1[:, 2]
                self.kappa = du_1[:, 3]

            l0 = r
            l1 = r * s
            l2 = 0.5 * sigma**2 * s**2

            # Since we transform t = T - t then we need to change the sign on u_t
            residual = -u_t - l0 * u[:, 0] + l1 * u_s + l2 * u_ss

            # Convert back to a (b,1) tensor from a (b,) tensor
            return residual[:, None]

    return BlackScholesWithResidual


@BlackScholesResidual
class BlackScholesSurrogate(PINN_Base):
    pass

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import Progbar
import tensorflow as tf


def train_lr_anneling(self, X, U, X_df, epochs, **kwargs):
    #optimizer = SGD(**kwargs)
    optimizer = Adam()

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    U = tf.convert_to_tensor(U, dtype=tf.float32)

    if self.use_differential_points:
        X_df = tf.convert_to_tensor(X_df, dtype=tf.float32)
    else:
        X_df = None

    progbar = Progbar(epochs)
    for i in range(epochs):
        with tf.GradientTape(persistent=True) as param_tape:
            loss, [loss_collocation, loss_residual,
                   loss_differential] = self._loss(X, U, X_df)

            loss_residual = loss_differential

        params = self.get_trainable_parameters()
        grads_collocation = param_tape.gradient(
            loss_collocation, params
        )
        grads_residual = param_tape.gradient(
            loss_residual, params
        )

        del param_tape

        # Find the maximum gradient for each of the two losses
        residual_max = 0
        for grad in grads_residual:
            residual_max = max(residual_max, tf.math.reduce_max(grad))

        collocation_max = 0
        for grad in grads_collocation:
            collocation_max = max(collocation_max, tf.math.reduce_max(grad))

        # Find the average for the collocation loss
        collocation_mean = tf.math.reduce_mean(
            tf.concat([
                tf.abs(tf.reshape(g, [-1])) for g in grads_collocation
            ], axis=0))

        l = residual_max / collocation_mean
        grads = [(l*g_c + g_r)
                 for g_c, g_r in zip(grads_collocation, grads_residual)]

        # Rescale the interior (residual) gradients
        optimizer.apply_gradients(zip(grads, params))

        progbar.update(i+1, [("loss", loss), ("reweight", l)])

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import Progbar
import tensorflow as tf


def train(self, X, U, X_df, epochs, **kwargs):
    optimizer = SGD(**kwargs)

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    U = tf.convert_to_tensor(U, dtype=tf.float32)

    if self.use_differential_points:
        X_df = tf.convert_to_tensor(X_df, dtype=tf.float32)
    else:
        X_df = None

    progbar = Progbar(epochs)
    for i in range(epochs):
        with tf.GradientTape() as param_tape:
            loss, [loss_collocation, loss_residual,
                   loss_differential] = self._loss(X, U, X_df)

            loss_residual = loss_residual + loss_differential

        params = self.get_trainable_parameters()
        grads_collocation = param_tape.gradient(
            loss_collocation, params
        )
        grads_residual = param_tape.gradient(
            loss_residual, params
        )

        # Find the maximum direction for each subset
        collocation_max = tf.math.reduce_max(tf.abs(grads_collocation))
        residual_max = tf.math.reduce_max(tf.abs(grads_residual))

        # Compute the ratio of the two directions
        l = residual_max / collocation_max

        # Rescale the interior (residual) gradients
        grads = l*grads_collocation + l*grads_residual

        optimizer.apply_gradients(zip(grads, params))

        progbar.update(i+1, [("loss", loss), ("reweight", l)])

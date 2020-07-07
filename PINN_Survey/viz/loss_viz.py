# Defines function decorators which turns classes of each architecture
# into one that can do 1d and 2d visualizations of the loss surface.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from imageio import imsave


def save_as_heightmap(fname, values):
    imsave(fname, values)


def linear_viz(model_viz, X, U, X_df, w0, w1, metric=None):
    fig, ax = plt.subplots()

    ts = np.linspace(-0.5, 1.5, 1000)
    metric_values = np.empty_like(ts)

    for i, t in enumerate(ts):
        loss = model_viz.linear_interpolate_error(X, U, w0, w1, t, metric)
        metric_values[i] = loss

    ax.plot(ts, metric_values)

    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Loss")

    return fig, ax, [ts, metric_values]


def viz_2d(model_viz, X, U, X_df, w0, d1, d2, n, levels):
    t1s = np.linspace(-1, 1, n)
    t2s = np.linspace(-1, 1, n)

    losses = np.empty((n, n))

    progbar = tf.keras.utils.Progbar(n * n)
    for i, t1 in enumerate(t1s):
        for j, t2 in enumerate(t2s):
            losses[i, j] = model_viz.interpolate_2d(
                X, U, X_df, w0, d1, d2, t1, t2)
            progbar.update(i * n + j + 1)

    fig, ax = plt.subplots()
    ax.contour(t1s, t2s, np.log(losses), levels=levels)

    return fig, ax, [t1s, t2s, losses]


def viz_2d_global_norm(model_viz, X, U, X_df, w0, n=100, levels=30):

    d1 = weights_unit_vector(w0)
    d2 = weights_unit_vector(w0)

    return viz_2d(model_viz, X, U, X_df, w0, d1, d2, n, levels)


def viz_2d_layer_norm(model_viz, X, U, X_df, w0, n=100, levels=30):

    d1 = weights_unit_vector_layerwise(w0)
    d2 = weights_unit_vector_layerwise(w0)

    return viz_2d(model_viz, X, U, X_df, w0, d1, d2, n, levels)


def viz_2d_svd(model_viz, X, U, X_df, w0, epoch_weights, n=100, levels=300):
    d1, d2, points = unit_vectors_SVD(epoch_weights)

    min_points = np.min(points, axis=0)
    max_points = np.max(points, axis=0)

    t1s = np.linspace(min_points[0] - 1.0, max_points[0] + 1.0, n)
    t2s = np.linspace(min_points[1] - 1.0, max_points[1] + 1.0, n)

    losses = np.empty((n, n))

    progbar = tf.keras.utils.Progbar(n * n)
    for i, t1 in enumerate(t1s):
        for j, t2 in enumerate(t2s):
            losses[i, j] = model_viz.interpolate_2d(
                X, U, X_df, w0, d1, d2, t1, t2)
            progbar.update(i * n + j + 1)

    fig, ax = plt.subplots()
    ax.contour(t1s, t2s, np.log(losses), levels=levels)

    ax.scatter(points[:, 0], points[:, 1])
    ax.scatter(points[0, 0], points[0, 1])
    ax.scatter(points[-1, 0], points[-1, 1])

    return fig, ax, [t1s, t2s, losses]


def unit_vectors_SVD(epoch_weights):

    assert(len(epoch_weights) > 1)
    w0 = epoch_weights[0]
    u0 = unwrap(w0)

    epoch_matrix = np.empty((len(epoch_weights) - 1, len(u0)))

    for i in range(epoch_matrix.shape[0]):
        w1 = epoch_weights[i + 1]
        u1 = unwrap(w1)

        delta = u0 - u1
        epoch_matrix[i, :] = delta[:]

    _, _, V = np.linalg.svd(epoch_matrix)
    assert(V.shape[1] == u0.shape[0])

    v1 = V[0, :]
    v2 = V[1, :]

    points = np.empty((epoch_matrix.shape[0], 2))
    for r in range(epoch_matrix.shape[0]):
        points[r, 0] = epoch_matrix[r, :] @ v1
        points[r, 1] = epoch_matrix[r, :] @ v2

    return wrap(v1, w0), wrap(v2, w0), points


def weights_unit_vector(w0):
    w_d = []
    length = 0
    for collection in w0:

        layers = []
        for w_l in collection:
            w = np.random.normal(size=w_l.shape)
            length += np.sum(w**2)
            layers.append(w)

        w_d.append(layers)

    # now normalize by the whole weight
    length = np.sqrt(length)
    return [[w / length for w in collection] for collection in w_d]


def weights_unit_vector_layerwise(w0):

    w_d = []
    for collection in w0:
        layers = []
        for w_l in collection:
            w = np.random.normal(size=w_l.shape)

            norm_w0 = np.sqrt(np.sum(w_l**2))

            norm_w1 = np.sqrt(np.sum(w**2))

            w = (w / norm_w1) * norm_w0

            layers.append(w)

        w_d.append(layers)

    return w_d


def unwrap(w0):
    flats = []

    for collection in w0:

        for layer in collection:
            flats.append(layer.flatten())

    return np.hstack(flats)


def wrap(flat_array, w0):

    start = 0
    w1 = []
    for collection in w0:

        new_collection = []
        for layer in collection:
            end = start + layer.size
            sub_layer = np.reshape(flat_array[start:end], layer.shape)
            new_collection.append(sub_layer)

            start = end

        w1.append(new_collection)

    return w1


def viz_base(cls):

    class viz_base_class(cls):

        def _init_NN_placeholder(self, layers):
            weights = []
            biases = []
            for l in range(len(layers) - 1):
                W = tf.placeholder(self.dtype, shape=[
                                   layers[l], layers[l + 1]])
                b = tf.placeholder(self.dtype, shape=[1, layers[l + 1]])
                weights.append(W)
                biases.append(b)

            return weights, biases

        def _init_params(self):
            self.weights, self.biases = self._init_NN_placeholder(self.layers)

        def _init_optimizers(self):
            # Since there are no optimizers
            pass

        @staticmethod
        def linear_interpolate_weights(w0, w1, t):

            assert(len(w0) == len(w1))
            w_t = []
            for collection_0, collection_1 in zip(w0, w1):

                collection_t = []
                for layer_idx in range(len(collection_0)):
                    w = (1 - t) * \
                        collection_0[layer_idx] + t * collection_1[layer_idx]
                    collection_t.append(w)

                w_t.append(collection_t)

            return w_t

        @staticmethod
        def scale_weights_2d(w0, d1, d2, t1, t2):
            # TODO: allow for n-d visualization?

            w_t = []
            for c0, c1, c2 in zip(w0, d1, d2):

                layers = []
                for l in range(len(c0)):
                    w = c0[l] + t1 * c1[l] + t2 * c2[l]
                    layers.append(w)

                w_t.append(layers)

            return w_t

        def get_input_dict(self, X, U, X_df):

            if X_df is not None:
                return {
                    self.X: X,
                    self.U: U,
                    self.X_df: X_df,
                }
            else:
                return {
                    self.X: X,
                    self.U: U,
                }

        @staticmethod
        def get_param_dict(w_p, w_v):

            return {
                tuple(weight_pl): tuple(weight_val)
                for weight_pl, weight_val in zip(w_p, w_v)
            }

        def run_with_weights(self, X, U, X_df, w_t, metrics):

            input_dict = self.get_input_dict(X, U, X_df)
            param_dict = self.get_param_dict(
                self.get_all_weight_variables(), w_t)

            feed_dict = {
                **input_dict,
                **param_dict
            }

            if metrics is not None:
                op = metrics
            else:
                op = self.loss

            return self.sess.run(op, feed_dict)

        def linear_interpolate_loss(self, X, U, X_df, w0, w1, t, metrics=None):
            w_t = self.linear_interpolate_weights(w0, w1, t)
            return self.run_with_weights(X, U, X_df, w_t, metrics)

        def interpolate_2d(self, X, U, X_df, w0, d1, d2, t1, t2, metrics=None):
            w_t = self.scale_weights_2d(w0, d1, d2, t1, t2)
            return self.run_with_weights(X, U, X_df, w_t, metrics)

    return viz_base_class


def viz_mesh(cls):

    cls_base = viz_base(cls)

    class viz_mesh_class(cls_base):

        def _init_params(self):
            self.weights_mesh, self.biases_mesh = self._init_NN_placeholder(
                self.layers_mesh)

            super()._init_params()

    return viz_mesh_class


def viz_domain_trainsformer(cls):

    def _init_params(self):

        # Two layer neural network
        transformer_shape = [self.input_dim, self.width, self.width]

        # The first encoder network
        self.weights_T1, self.biases_T1 = self._init_NN_placeholder(
            transformer_shape)
        # The second encoder network
        self.weights_T2, self.biases_T2 = self._init_NN_placeholder(
            transformer_shape)

        # The normal "forward" network is initialized by parent
        super()._init_params()

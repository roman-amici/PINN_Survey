# Defines function decorators which turns classes of each architecture
# into one that can do 1d and 2d visualizations of the loss surface.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from imageio import imsave
from PINN_Base import util


def save_as_heightmap(fname, values):
    '''
    Save a grid of values as a greyscale image
    readable by paraview.
    '''

    imsave(fname, values)


def linear_viz(model_viz, X, U, X_df, w0, w1, metric=None):
    '''
    Visualize the loss function by linearly interpolating between
    w0 and w1.
    '''

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


def viz_2d(model_viz, X, U, X_df, w0, d1, d2, n, metrics=None, min_vals=[-1, -1], max_vals=[1, 1]):
    '''
    Compute a grid of metric values by interpolating the two directions, (d1,d2)
    using the formula
    w0 + t_1*d1 + t_2*d2

    If no metric is given, the "loss" metric will be used.

    model_viz should be a PINN_vase model decorated with viz_base
    '''

    t1s = np.linspace(min_vals[0], max_vals[0], n)
    t2s = np.linspace(min_vals[1], max_vals[1], n)

    if metrics is None:
        metrics = [model_viz.loss]

    metric_values = [
        np.empty((n, n)) for _ in metrics
    ]

    progbar = tf.keras.utils.Progbar(n * n)
    for i, t1 in enumerate(t1s):
        for j, t2 in enumerate(t2s):
            results = model_viz.interpolate_2d(
                X, U, X_df, w0, d1, d2, t1, t2, metrics)
            for m, result in enumerate(results):
                metric_values[m][i, j] = result
            progbar.update(i * n + j + 1)

    if len(metric_values) == 1:
        return t1s, t2s, metric_values[0]
    else:
        return t1s, t2s, metric_values


def viz_2d_row_norm(model_viz, X, U, X_df, w0, n=100, metrics=None):
    '''
    Compute a grid of values using weight directions normalized by each
    row in the weight matrix for each layer.
    '''

    d1 = weights_unit_vector_rowwise(w0)
    d2 = weights_unit_vector_rowwise(w0)

    return viz_2d(model_viz, X, U, X_df, w0, d1, d2, n, metrics)


def viz_2d_global_norm(model_viz, X, U, X_df, w0, n=100, metrics=None):
    '''
    Compute a grid of values using weight directions normalized to be
    unit vectors in the total weight space.
    '''

    d1 = weights_unit_vector(w0)
    d2 = weights_unit_vector(w0)

    return viz_2d(model_viz, X, U, X_df, w0, d1, d2, n, metrics)


def viz_2d_layer_norm(model_viz, X, U, X_df, w0, n=100, metrics=None):
    '''
    Compute a grid of values using weight directions normalized by the norm
    of each weight matrix in a layer.
    '''

    d1 = weights_unit_vector_layerwise(w0)
    d2 = weights_unit_vector_layerwise(w0)

    return viz_2d(model_viz, X, U, X_df, w0, d1, d2, n, metrics)


def make_contour_svd(ax, t1s, t2s, values, points, singular_values, levels=30):
    '''
    Construct a contour plot of the metric values with the
    path taken by the optimizer in that path.

    Use in conjunction with viz_2d_svd to get appropriate values.
    '''

    sv = singular_values**2 / (np.sum(singular_values**2))
    sv_1 = sv[0] * 100
    sv_2 = sv[1] * 100

    ax.set_ylabel(f"1st PCA component {sv_1:3.3}%")
    ax.set_xlabel(f"2nd PCA component {sv_2:3.3}%")

    ax.contour(t2s, t1s, np.log(values), levels=levels)

    ax.scatter(points[:, 1], points[:, 0], zorder=2)
    ax.scatter(points[0, 1], points[0, 0], c=["k"], zorder=2)
    ax.scatter(points[-1, 1], points[-1, 0], marker="X", c=["r"], zorder=2)


def viz_2d_svd(model_viz, X, U, X_df, w0, epoch_weights, n=100, metrics=None):
    '''
    Used to visualize the trajectory of training in terms of the given loss metrics.

    Parameters:
        epoch_weights - A list of weight values at each training epoch.

    '''

    d1, d2, points, singular_values = unit_vectors_SVD(epoch_weights)

    min_vals = np.min(points, axis=0) - 1.0
    max_vals = np.max(points, axis=0) + 1.0

    t1s, t2s, metric_values = viz_2d(
        model_viz, X, U, X_df, w0, d1, d2, n, metrics, min_vals, max_vals)

    return t1s, t2s, metric_values, points, singular_values


def unit_vectors_SVD(epoch_weights):
    '''
    Uses the top two singular vectors built from the matrix of (relative)
    descent directions.
    '''

    assert(len(epoch_weights) > 1)
    wn = epoch_weights[-1]
    un = unwrap(wn)

    # Construct a matrix of relative descent directions relative to the
    # final vector of weights. I.e. each row is
    # w_i - w_final
    epoch_matrix = np.empty((len(epoch_weights) - 1, len(un)))

    for i in range(epoch_matrix.shape[0]):
        w1 = epoch_weights[i]
        u1 = unwrap(w1)

        delta = u1 - un
        epoch_matrix[i, :] = delta[:]

    # Take the top two singular values
    _, singular_vals, V = np.linalg.svd(epoch_matrix)
    assert(V.shape[1] == un.shape[0])

    v1 = V[0, :]
    v2 = V[1, :]

    t1 = un @ v1
    t2 = un @ v2

    # Project the weight values onto the descent directions
    points = np.empty((len(epoch_weights), 2))
    for r in range(len(epoch_weights)):
        w1 = epoch_weights[r]
        u1 = unwrap(w1)
        # We need to subtract the projection onto w_final to fix the origin to be w_final.
        points[r, 0] = (u1 @ v1) - t1
        points[r, 1] = (u1 @ v2) - t2

    return wrap(v1, wn), wrap(v2, wn), points, singular_vals


def weights_unit_vector(w0):
    '''
    Construct a unit vector with the same (unwrapped and the rewrapped)
    shape as w0
    '''

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


def weights_unit_vector_rowwise(w0):
    '''
    Construct a direction vector with the same shape as w0.
    Each 'row' in each weight matrix of each layer is rescaled
    so that it has the same magnitude as the same row in w0.
    '''

    w_d = []
    for collection in w0:
        layers = []
        for w_l in collection:
            w_new = np.empty_like(w_l)

            for row in range(w_new.shape[0]):
                w = np.random.normal(size=w_l.shape[1])

                norm_w0 = np.linalg.norm(w_l[row, :])
                norm_w1 = np.linalg.norm(w)

                w = (w / norm_w1) * norm_w0
                w_new[row, :] = w

            layers.append(w_new)
        w_d.append(layers)

    return w_d


def weights_unit_vector_layerwise(w0):
    '''
    Construct a direction vector with the same shape as w0.
    Each  weight matrix of each layer is rescaled
    so that it has the same Frobenious as the same matrix in w0.
    '''

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
    '''
    Weights returned from get_all_weights() is a list of list of matrices.
    The first list is a list of weight categories. For PINN_Base this weights and biases.
    Each of these contains a list of matrices corresponding to a neural network layer.

    This function flattens this list of list of matrices into a single, flat array
    '''

    flats = []

    for collection in w0:

        for layer in collection:
            flats.append(layer.flatten())

    return np.hstack(flats)


def wrap(flat_array, w0):
    '''
    Wraps the given 1D array to have the same list of list of 
    matrices structure as w0.
    '''

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
    '''
    A decorator to add visualization functionality to a PINN Model.
    This version of the model cannot be trained; instead weights are
    passed in as a parameter in sess.run calls.
    '''

    class viz_base_class(cls):

        def _init_NN_placeholder(self, layers):
            '''
            Instead of being a tf.Variable, we make these tf.Placeholders
            so that we can specify weight values at runtime.
            '''

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
            # Since there are no variables, there are no optimizers
            pass

        @staticmethod
        def linear_interpolate_weights(w0, w1, t):
            '''
            Compute (1-t) * w0 + t * w1 and return the result.
            '''

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
            '''
            Compute w0 + t1 * d1 + t2 * d2 and return the result.
            '''

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

        def matvec_with_weights(self, v, X, U, X_df, w_t):
            '''
            Perform a Hessian matrix vector mutliply Hv but with
            weights specified with w_t rather than being variables.
            '''

            input_dict = self.get_input_dict(X, U, X_df)
            input_dict[self.hessian_vector] = v

            param_dict = self.get_param_dict(
                self.get_all_weight_variables(), w_t)

            feed_dict = {
                **input_dict,
                **param_dict
            }

            return util.unwrap(self.sess.run(self.hessian_matvec, feed_dict))

        def make_matvec_fn_with_weights(self, X, U, X_df, w0, d1, d2, t1, t2):
            '''
            Creates a closure which encapsulates the Hessian matvec operation
            which can be passed to a generic Lanczos iteration algorithm
            '''

            w_t = self.scale_weights_2d(w0, d1, d2, t1, t2)

            def matvec_fn(v):
                return self.matvec_with_weights(v, X, U, X_df, w_t)

            return matvec_fn

    return viz_base_class

# Each architecture has a slightly different set of params
# and thus needs its own class to convert it to params.
# Consider during this into a property of the architectures themselves.


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

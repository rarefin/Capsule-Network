import numpy as np
import tensorflow as tf


class CapsuleNetwork:
    def margin_loss(self, y, y_predicted):
        """
            :param y: a vector of size num_class containing the 1 in correct class position
            :param y_predicted: a vector containing the probability of each class..
            These are the normalized length of output layer capsule
            :return: a scalar value of loss
            """
        m_plus = 0.9  # for positive class probability should be at least 0.9
        m_minus = 0.1
        lambd = 0.5

        loss = tf.reduce_sum(y * tf.square(tf.maximum(0.0, (m_plus - y_predicted))) + lambd * (1 - y)
                             * tf.square(tf.maximum(0.0, (y_predicted - m_minus))), axis=1)
        loss = tf.reduce_mean(loss)
        return loss

    def squash(self, tensor):
        """Squashing function
        :param tensor: A tensor with shape [batch_size, 1, num_caps, caps_dim, 1] or [batch_size, num_caps, caps_dim, 1].
        :return: A tensor with the same shape as vector but squashed in 'caps_dim' dimension.
        """
        vec_squared_norm = tf.reduce_sum(tf.square(tensor), -2, keep_dims=True)
        scalar_factor = tf.sqrt(vec_squared_norm) / (1 + vec_squared_norm)
        vec_squashed = scalar_factor * tensor

        return vec_squashed

    def PrimaryCapsules(self, inputs, num_capsules, dim_capsule, kernel_size, strides, padding):
        """
            :param inputs: input tensor
            :param num_capsules: number of capsules to create
            :param dim_capsule: dimension of capsule
            :param kernel_size
            :param strides
            :param padding

            :return: return primary capsules
        """
        capsules = tf.layers.conv2d(inputs=inputs, filters=num_capsules * dim_capsule, kernel_size=kernel_size,
                                    strides=strides, padding=padding)
        capsules = tf.reshape(capsules, shape=[-1, 32, 36, dim_capsule, 1])

        return capsules

    def DigitCapsules(self, batch_size, primaryCaps, no_of_iterations=3):
        """
                :param inputs: input tensor
                :param num_capsules: number of capsules to create
                :param dim_capsule: dimension of capsule
                :param kernel_size
                :param strides
                :param padding

                :return: return primary capsules
            """
        weight_matrix = tf.Variable(np.random.normal(size=[1, 32, 1, 16, 8], scale=0.01), dtype=tf.float32)
        weight_matrix = tf.tile(weight_matrix, multiples=[batch_size, 1, 36, 1, 1]) # [batch_size, 32, 36, 16, 8]

        prediction_vectors = tf.matmul(weight_matrix, primaryCaps) # [batch_size, 32, 36, 16, 1]
        prediction_vectors = tf.reshape(tensor=prediction_vectors, shape=[-1, 1, 1152, 16, 1]) # [batch_size, 1, 1152, 16, 1]
        prediction_vectors = tf.tile(input=prediction_vectors, multiples=[1, 10, 1, 1, 1]) # [batch_size, 10, 1152, 16, 1]

        prediction_vectors_grad_stopped = tf.stop_gradient(input=prediction_vectors)

        log_priors = tf.Variable(np.zeros(shape=[1, 10, 1152, 1, 1], dtype=np.float32), dtype=tf.float32) # [1, 10, 1152, 1, 1]

        # Routing algorithm
        for r in range(no_of_iterations):
            with tf.variable_scope('routing_iter_' + str(r)):
                coupling_coefficients = tf.nn.softmax(log_priors, dim=1)
                weighted_sum = tf.reduce_sum(tf.multiply(coupling_coefficients, prediction_vectors_grad_stopped), axis=2,
                                             keep_dims=True)  # [batch_size, 10, 1, 16, 1]

                activation_capsules = self.squash(weighted_sum)  # [batch_size, 10, 1, 16, 1]
                activation_capsules_tiled = tf.tile(activation_capsules,
                                                    [1, 1, 1152, 1, 1])  # [batch_size, 10, 1152, 16, 1]
                u_produce_v = tf.matmul(prediction_vectors, activation_capsules_tiled,
                                        transpose_a=True)  # [batch_size, 10, 1152, 1, 1]
                log_priors += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)

        with tf.variable_scope('activation_capsules'):
            coupling_coefficients = tf.nn.softmax(log_priors, dim=1)
            weighted_sum = tf.reduce_sum(tf.multiply(coupling_coefficients, prediction_vectors), axis=2, keep_dims=True)  # [batch_size, 10, 1, 16, 1]
            activation_capsules = self.squash(weighted_sum)  # [batch_size, 10, 1, 16, 1]
            activation_capsules = tf.reshape(tensor=activation_capsules, shape=[-1, 10, 16]) # [batch_size, 10, 16]

        return activation_capsules

    def create_CapsuleNetwork(self, X, parameters=None, reuse=True, isTraining=True):
        """
                :param X: input tensor
                :param parameters: parameters of capsule network

                :return: lengths of digitcaps layer capsules
            """

        # Convolution Layer with 256 filters and a kernel size of 5
        batch_size = tf.shape(X)[0]
        with tf.variable_scope('PrimaryConv'):
            conv1 = tf.layers.conv2d(inputs=X, filters=256, kernel_size=9, strides=1, padding="valid",
                                     activation=tf.nn.relu)

        with tf.variable_scope('PrimaryCaps'):
            primaryCaps = self.PrimaryCapsules(inputs=conv1, num_capsules=32, dim_capsule=8, kernel_size=9,
                                          strides=2, padding="valid")
            primaryCaps = self.squash(primaryCaps)

        with tf.variable_scope('DigitCaps'):
            digit_caps = self.DigitCapsules(batch_size, primaryCaps) # [batch_size, 10, 16]
            digit_caps_lengths = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=2))

        return digit_caps_lengths


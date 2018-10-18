import tensorflow as tf
import numpy as np


class ValueEstimator:
    def __init__(self, scope, state_size, action_size, isDiscrete, conv_sizes, fc_sizes,
                 learning_rate=0.01, normalize_input=True):

        self.scope = scope
        self.state_size = state_size  # observation(input) dimension
        self.action_size = action_size  # action(output)
        self.conv_sizes = conv_sizes  # FC layer unit sizes - output layer
        self.fc_sizes = fc_sizes

        self.learning_rate = learning_rate
        self.normalize_input = normalize_input
        self.isDiscrete = isDiscrete  # discrete/continuous actions

        self.state = None
        self.action = None
        self.target = None
        self.value_est = None
        self.selected_value_est = None
        self.train_op = None
        self.loss = None

        # Build model
        self.build_model(self.scope)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        value_est = sess.run(self.value_est, feed_dict={self.state: state})
        return value_est

    def update(self, states, actions, targets, sess=None):
        sess = sess or tf.get_default_session()
        _, loss_val, selected_value_est = sess.run([self.train_op, self.loss, self.selected_value_est],
                                                   feed_dict={self.state: states, self.action: actions,
                                                              self.target: targets})

        return loss_val

    def build_model(self, scope="value_estimator"):
        with tf.variable_scope(scope):
            # Define place holders
            self.state = tf.placeholder(shape=[None, ] + self.state_size, name="state", dtype=tf.uint8)
            inputs = self.state
            inputs = tf.to_float(inputs) / 255.0

            self.target = tf.placeholder(shape=[None], name="target", dtype=tf.float32)
            self.action = tf.placeholder(shape=[None], name="action", dtype=tf.int32)

            # Define initializer
            initializer = None # tf.contrib.layers.xavier_initializer()

            # Q-function (action-value function)
            with tf.name_scope('model'):
                nLayers = 1

                # Conv2D layers
                for filter_size, kernel_size, stride in self.conv_sizes:
                    inputs = tf.layers.conv2d(inputs=inputs, filters=filter_size,
                                              kernel_size=kernel_size,
                                              strides=stride,
                                              activation=tf.nn.relu,
                                              name="conv2d_" + str(nLayers))
                    nLayers += 1

                # Fully connected layers
                flattened = tf.layers.flatten(inputs=inputs, name="flattened")
                fc = flattened

                n_fcs = 0
                for fc_size in self.fc_sizes:
                    fc = tf.layers.dense(inputs=fc, units=fc_size, activation=tf.nn.relu, name="f1_" + str(n_fcs))
                    n_fcs += 1

                self.value_est = tf.layers.dense(inputs=fc, units=self.action_size, activation=None, name="est")

                # Get value estimation of selected action
                self.selected_value_est = tf.reduce_sum(self.value_est * tf.one_hot(self.action, self.action_size),
                                                        axis=1)

            # Loss function
            with tf.name_scope("loss"):
                self.loss = tf.nn.l2_loss(self.selected_value_est - self.target)

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class StateProcessor:
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """

    def __init__(self, input_shape, output_shape):
        # Build the Tensorflow graph
        self.output_shape = output_shape
        self.input_shape = input_shape
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=self.input_shape, dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, self.output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, state, sess=None):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """

        sess = sess or tf.get_default_session()
        return sess.run(self.output, {self.input_state: state})

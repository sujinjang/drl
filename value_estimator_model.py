import tensorflow as tf
import numpy as np


class ValueEstimator:
    def __init__(self, n_state, isDiscrete, layer_sizes, activation_fn='relu', learning_rate=0.01):
        self.n_state = n_state  # observation(input) dimension
        self.layer_sizes = layer_sizes  # FC layer unit sizes - output layer

        if activation_fn == "relu":
            self.activation_fn = tf.nn.relu
        elif activation_fn == 'tanh':
            self.activation_fn = tf.tanh
        else:
            self.activation_fn = None

        self.learning_rate = learning_rate
        self.isDiscrete = isDiscrete  # discrete/continuous actions

        self.state = None
        self.target = None
        self.value_est = None
        self.train_op = None
        self.loss = None

        # Build model
        self.build_model()

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        value_est = sess.run(self.value_est, feed_dict={self.state: state})
        return value_est

    def update(self, states, targets, sess=None):
        sess = sess or tf.get_default_session()
        _, loss_val, value_est = sess.run([self.train_op, self.loss, self.value_est],
                                          feed_dict={self.state: states, self.target: targets})

        print("-----------Value-----------")
        print("loss_val: {}".format(loss_val))
        print("mean_value_est: {} ".format(np.mean(value_est)))
        print("std_value_est: {}".format(np.std(value_est)))

    def build_model(self, scope="value_estimator"):
        with tf.name_scope(scope):
            # Define place holders
            self.state = tf.placeholder(shape=[None, self.n_state], name="state", dtype=tf.float32)
            self.target = tf.placeholder(shape=[None], name="target", dtype=tf.float32)

            initializer = tf.contrib.layers.xavier_initializer()
            # initializer = None

            # Forward policy pass, pi(a|s)
            with tf.name_scope('model'):
                out = self.state
                nLayers = 1
                for size in self.layer_sizes:
                    out = tf.layers.dense(inputs=out, units=size, activation=self.activation_fn,
                                          bias_initializer=initializer,
                                          kernel_initializer=initializer,
                                          name="fc_" + str(nLayers) + "_" + scope)
                    nLayers += 1

                self.value_est = tf.layers.dense(inputs=out, units=1, activation=None,
                                                 bias_initializer=initializer,
                                                 kernel_initializer=initializer,
                                                 name="value_est")

                # Loss function
            with tf.name_scope("loss"):
                self.loss = tf.nn.l2_loss(self.value_est - self.target)

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

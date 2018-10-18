import tensorflow as tf
import numpy as np


class PolicyEstimator:
    def __init__(self, n_state, n_action, isDiscrete, layer_sizes, isAdaptiveStd=True, std_layer_sizes=[32, 32],
                 activation_fn='tanh', learning_rate=0.01):
        self.n_state = n_state  # observation(input) dimension
        self.n_action = n_action  # action(output) dimension
        self.layer_sizes = layer_sizes  # FC layer unit sizes - output layer
        self.isAdaptiveStd = isAdaptiveStd
        self.std_layer_sizes = std_layer_sizes

        if activation_fn == "relu":
            self.activation_fn = tf.nn.relu
        elif activation_fn == 'tanh':
            self.activation_fn = tf.tanh
        else:
            self.activation_fn = None

        self.learning_rate = learning_rate
        self.isDiscrete = isDiscrete  # discrete/continuous actions

        self.state = None
        self.action = None
        self.advantage = None  # for policy estimator
        self.sampled_action = None
        self.logits = None
        self.log_prob = None
        self.mean = None
        self.log_std = None
        self.std = None
        self.zs = None
        self.loss = None
        self.norm_dist = None
        self.train_op = None

        # Build model
        self.build_model()

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        action = sess.run(self.sampled_action, feed_dict={self.state: [state]})
        return action

    def update(self, states, actions, advantages, sess=None):
        sess = sess or tf.get_default_session()
        _, loss_val, log_prob = sess.run([self.train_op, self.loss, self.log_prob],
                                         feed_dict={
                                             self.state: states,
                                             self.action: actions,
                                             self.advantage: advantages})

        print("-----------Policy-----------")
        print("loss_val: {}".format(loss_val))
        print("mean_log_prob: {} ".format(np.mean(log_prob)))
        print("std_log_prob: {}".format(np.std(log_prob)))

    def build_model(self, scope="policy_estimator"):
        with tf.name_scope(scope):
            # Define place holders
            self.state = tf.placeholder(shape=[None, self.n_state], name="state", dtype=tf.float32)  # state variable

            if self.isDiscrete:
                self.action = tf.placeholder(shape=[None], name="action", dtype=tf.int32)
            else:
                self.action = tf.placeholder(shape=[None, self.n_action], name="action", dtype=tf.float32)

            self.advantage = tf.placeholder(shape=[None], name="advantage", dtype=tf.float32)

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

            # Compute Log[pi(a|s)]
            if self.isDiscrete:
                with tf.name_scope('output'):
                    self.logits = tf.layers.dense(inputs=out, units=self.n_action, activation=None,
                                                  bias_initializer=initializer,
                                                  kernel_initializer=initializer,
                                                  name="logit")

                    # Sample an action from the stochastic policy
                    self.sampled_action = tf.multinomial(logits=self.logits, num_samples=1)
                    self.sampled_action = tf.reshape(self.sampled_action, [-1])

                    # Negative log-probability (cross-entropy)
                    self.log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                   labels=self.action)

            else:
                # self.mean = logits
                # self.logstd = tf.get_variable(name="std", shape=[1, self.n_action], dtype=tf.float32,
                #                          initializer=tf.zeros_initializer())
                # self.std = tf.exp(self.logstd)
                #
                # self.sampled_action = tf.random_normal(shape=tf.shape(self.mean), mean=self.mean, stddev=self.std)
                # self.sampled_action = tf.reshape(self.sampled_action, [-1, self.n_action])
                #
                # # Equivalent to negative-log-likelihood-fication (max log = min (-log) for a computational reliability)
                # self.norm_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.mean, scale_diag=self.std)
                # self.logProb = -self.norm_dist.log_prob(self.action)

                with tf.name_scope('output'):
                    self.mean = tf.layers.dense(inputs=out, units=self.n_action, activation=None,
                                                bias_initializer=initializer,
                                                kernel_initializer=initializer,
                                                name="mean")

                if self.isAdaptiveStd:
                    self.log_std = self.state
                    nLayers = 1
                    for size in self.std_layer_sizes:
                        self.log_std = tf.layers.dense(inputs=self.log_std, units=size, activation=self.activation_fn,
                                                       bias_initializer=initializer,
                                                       kernel_initializer=initializer,
                                                       name="fc_log_std_" + str(nLayers) + "_" + scope)
                        nLayers += 1

                    self.log_std = tf.layers.dense(inputs=self.log_std, units=self.n_action,
                                                   activation=self.activation_fn,
                                                   bias_initializer=initializer,
                                                   kernel_initializer=initializer,
                                                   name="log_std")

                else:
                    self.log_std = tf.Variable(initial_value=tf.zeros([1, self.n_action]),
                                               trainable=True, name="log_std")

                # Get sampled action
                self.sampled_action = tf.random_normal(shape=tf.shape(self.mean))
                self.sampled_action = self.sampled_action * tf.exp(self.log_std) + self.mean

                # Get Log Likelihood
                self.zs = (self.action - self.mean) / tf.exp(self.log_std)
                log_norm_t1 = -0.5 * tf.constant([np.pi * 2])
                log_norm_t2 = -tf.reduce_sum(self.log_std, axis=1)
                log_norm_t3 = -0.5 * tf.reduce_sum(tf.square(self.zs), axis=1)
                self.log_prob = -(log_norm_t1 + log_norm_t2 + log_norm_t3)

            # Loss function
            with tf.name_scope("loss"):
                self.loss = tf.reduce_sum(tf.multiply(self.log_prob, self.advantage))

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

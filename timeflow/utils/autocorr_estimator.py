import tensorflow as tf
import numpy as np

class AutoCorrEstimator():

    def __init__(self, time_input, signal_input):
        self.time_input = time_input
        self.signal_input = signal_input
        with tf.variable_scope('phi'):
            self.phi_tf = tf.Variable(tf.random_uniform([1], maxval=.8, minval=.3))
        with tf.variable_scope('sigma'):
            self.sigma_tf = tf.Variable(tf.random_uniform([1], maxval=8., minval=.3))
        self.next_signal, self.prev_signal = tf.unpack(self.signal_input, axis=1)
        # Extending the dimensions of both the vectors
        self.next_signal = tf.expand_dims(self.next_signal, axis=1)
        self.prev_signal = tf.expand_dims(self.prev_signal, axis=1)
        # Packing all the three tensors for computation
        self.input_ = tf.pack([self.time_input, self.next_signal, self.prev_signal], axis=2)

    def generate_log_loss(self):
        nu = tf.mul(self.sigma_tf, tf.sqrt(tf.sub(1., tf.pow(self.phi_tf, tf.mul(2., self.time_input)))))
        e = tf.sub(tf.mul(tf.pow(self.phi_tf, self.time_input), self.prev_signal), self.next_signal)
        nu_sq = tf.pow(nu, 2.)
        e_sq = tf.pow(e, 2.)
        # Calculating all steps of LL
        log_lik = tf.log(tf.mul(2.51, nu)) + tf.div(e_sq, tf.mul(2., nu_sq))
        return tf.reduce_sum(log_lik)

    def minimize_log_loss(self, time_data, signal_data):
        prev_log_loss = 1e8
        log_likelihood = self.generate_log_loss()
        #Initialize the Session
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(log_likelihood)
        sess=tf.InteractiveSession()
        tf.global_variables_initializer().run()
        #Calculating the log_loss for the first time with standard inputs
        _, computed_log_lik = sess.run([train_step, log_likelihood],
                                    feed_dict={self.time_input:time_data,
                                    self.signal_input:signal_data})
        while (prev_log_loss - computed_log_lik > 1e-7):
            prev_log_loss = computed_log_lik
            _, computed_log_lik = sess.run([train_step, log_likelihood],
                                        feed_dict={self.time_input:time_data,
                                        self.signal_input:signal_data})
        #Computing the value of phi and sigma
        phi_value, sigma_value = sess.run([self.phi_tf[0], self.sigma_tf[0]])
        return phi_value, sigma_value

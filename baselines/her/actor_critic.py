import tensorflow as tf
from baselines.her.util import store_args, nn
from baselines.her.gumbel import GumbelSoftmax

class ActorCritic: 
    @store_args
    def __init__(self, inputs_tf, action_type, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal的 (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        if action_type == "discrete":
            u = self.u_tf - 0.5
        else:
            u = self.u_tf

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            if action_type == "continuous":
                self.pi_tf = self.deter_pi_tf = self.max_u * tf.tanh(nn(
                    input_pi, [self.hidden] * self.layers + [self.dimu]))
            elif action_type == "discrete":
                # now a distribution.
                pi_nn = tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu]))
                pi_dist = GumbelSoftmax(1.0, tf.math.log(tf.nn.softmax(pi_nn)), dtype=tf.float32)
                pi_dist_soft = pi_dist.sample()
                pi_dist_hard = pi_dist.convert_to_one_hot(pi_dist_soft)
                pi_sample = pi_dist_soft - tf.stop_gradient(pi_dist_soft) + pi_dist_hard
                self.deter_pi_tf = pi_dist.mode()
                if self.net_type == "target":
                    self.pi_tf = pi_dist.mode()
                else:
                    self.pi_tf = pi_sample
            else:
                raise ValueError("Unsupported action type " + action_type)
        
        with tf.variable_scope('Q'):
            # for policy training
            if action_type == "continuous":
                input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            elif action_type == "discrete":
                # pi_dist_soft = pi_dist.sample()
                # pi_dist_hard = pi_dist.convert_to_one_hot(pi_dist_soft)
                # pi_sample = pi_dist_soft - tf.stop_gradient(pi_dist_soft) + pi_dist_hard
                centered_pi_tf = self.pi_tf - 0.5
                input_Q = tf.concat(axis=1, values=[o, g, centered_pi_tf])
                # input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf])
            else:
                raise ValueError("Unsupported action type " + action_type)

            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, u / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True) 
        with tf.variable_scope('gradient_Q_a'):
            self.gradient_Q_a = tf.gradients(self.Q_tf, self.u_tf)

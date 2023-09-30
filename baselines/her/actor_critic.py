import tensorflow as tf
from baselines.her.util import store_args, nn
from baselines.her.gumbel import GumbelSoftmax
from baselines.her.gumbel import GumbelSoftmax

class ActorCritic: 
    @store_args
    def __init__(self, inputs_tf, action_type, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                noise_std=0.2, noise_clip=0.5, truncate=False, **kwargs):
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
        # only used for target networks.
        self.success_tf = tf.placeholder(tf.float32, shape=(None, 1))

        if action_type == "discrete":
            u = self.u_tf - 0.5
        else:
            u = self.u_tf

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
                if self.noise_std != 0:
                    epsilon = tf.random.normal(shape=tf.shape(self.pi_tf), mean=0, stddev=noise_std)
                    epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
                    smoothed_pi_tf = tf.clip_by_value(self.pi_tf + epsilon, -self.max_u, self.max_u)
                    if truncate:
                        self.pi_tf = (1 - self.success_tf) * smoothed_pi_tf + self.success_tf * self.deter_pi_tf
                    else:
                        self.pi_tf = smoothed_pi_tf
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
            with tf.variable_scope('Q1'):
                # for policy training
                if action_type == "continuous":
                    input_Q1_pi = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
                elif action_type == "discrete":
                    centered_pi_tf = self.pi_tf - 0.5
                    input_Q1_pi = tf.concat(axis=1, values=[o, g, centered_pi_tf])
                else:
                    raise ValueError("Unsupported action type " + action_type)

                self.Q1_pi_tf = nn(input_Q1_pi, [self.hidden] * self.layers + [1])
                # for critic training
                input_Q1 = tf.concat(axis=1, values=[o, g, u / self.max_u])
                self._input_Q1 = input_Q1  # exposed for tests
                self.Q1_tf = nn(input_Q1, [self.hidden] * self.layers + [1], reuse=True)
            # with tf.variable_scope('gradient_Q1_a'):
            #     self.gradient_Q1_a = tf.gradients(self.Q1_tf, self.u_tf)


            # Second Critic (Q2)
            with tf.variable_scope('Q2'):
                # for policy training
                if action_type == "continuous":
                    input_Q2_pi = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
                elif action_type == "discrete":
                    centered_pi_tf = self.pi_tf - 0.5
                    input_Q2_pi = tf.concat(axis=1, values=[o, g, centered_pi_tf])
                else:
                    raise ValueError("Unsupported action type " + action_type)

                self.Q2_pi_tf = nn(input_Q2_pi, [self.hidden] * self.layers + [1])

                # for critic training
                input_Q2 = tf.concat(axis=1, values=[o, g, u / self.max_u])
                self._input_Q2 = input_Q2  # exposed for tests
                self.Q2_tf = nn(input_Q2, [self.hidden] * self.layers + [1], reuse=True)
            
            self.Q_pi_tf = tf.minimum(self.Q1_pi_tf, self.Q2_pi_tf)
            self.Q_tf = tf.minimum(self.Q1_tf, self.Q2_tf)

        self.IQR_pi_tf = self.Q1_pi_tf
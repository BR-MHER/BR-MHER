from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch,
    convert_episode_to_batch_major, g_to_ag, env_action_type)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util
from baselines.her.dynamics import ForwardDynamics, ForwardDynamicsNumpy
import time
from collections import defaultdict

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(self, action_space, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 gamma, samplers, mode, n_step, cor_rate, dynamic_batchsize, dynamic_init, alpha, lamb,
                 grad_to_lb=False, reg_pos=False, reg_ratio=0.1, reuse=False, anchor_offset_limit=np.inf,
                 grad_clip_value=-1, et=False, temperature=1.0, clip_power=False, clip_exp=10000, exp_degree=3, success_scale=1, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
            n_steps: number of steps to boostrap the return
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        print(anchor_offset_limit, "LIMITIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        self.create_actor_critic = import_function(self.network_class)
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']
        self.action_type = env_action_type(self.action_space)

        # Prepare staging area for feeding data to the model. save data for her process
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        stage_shapes['success_within_n_steps'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key]) for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size # buffer_size % rollout_batch_size should be zero

        self.basic_info = {'anchor_offset_limit': anchor_offset_limit, "et": et}
        self.her_info = {'get_Q_pi': self.get_Q_pi, 'gamma': self.gamma, 'get_Q': self.get_Q}
        self.her_info.update(self.basic_info)

        if self.mode == 'dynamic':
            sampler = self.samplers[self.mode]
            self.info = {
                'nstep':self.n_step,
                'gamma':self.gamma,
                'get_Q_pi':self.get_Q_pi,
                'get_Q': self.get_Q,
                'dynamic_model':self.dynamic_model,
                'get_action_and_Q_pi': self.get_action_and_Q_pi,
                'alpha':self.alpha,
                'use_dynamic_nstep':True
            }
        elif self.mode == 'lambda':
            sampler = self.samplers[self.mode]
            self.info = {
                'nstep':self.n_step,
                'gamma':self.gamma,
                'get_Q_pi':self.get_Q_pi,
                'get_Q':self.get_Q,
                'lamb':self.lamb,
                'use_lambda_target':True
            }
        elif self.mode == 'mix':
            sampler = self.samplers[self.mode]
            self.info = {
                'nstep':self.n_step,
                'gamma':self.gamma,
                'get_Q_pi':self.get_Q_pi,
                'get_Q':self.get_Q,
                'alpha':self.alpha,
            }
        elif self.mode == "correct":
            sampler = self.samplers[self.mode]
            self.info = {
                'nstep':self.n_step,
                'gamma':self.gamma,
                'use_correct':True,
                'get_Q_pi':self.get_Q_pi,
                'get_Q':self.get_Q,
                'cor_rate':self.cor_rate
            }
        elif self.mode in ["nstep", "tnstep"]:
            sampler = self.samplers[self.mode]
            self.info = {
                'nstep':self.n_step,
                'gamma':self.gamma,
                'use_nstep':True,
                'get_Q_pi':self.get_Q_pi,
                'get_Q': self.get_Q
            }
        else:
            sampler = self.samplers['her']
            self.info = self.her_info
        self.info.update(self.basic_info)
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, sampler, self.info)
        
        # self.bias_stats = defaultdict(list)
        self.loss_stats = defaultdict(list)
        self.grad_stats = defaultdict(list)
        self.mix_stats = defaultdict(list)
    
    def get_bias(self):
        pass
        # stats_length = [len(v) for v in self.bias_stats.values()]
        # assert stats_length[0] == np.mean(stats_length), "the bias information is incorrect"
        # logs = [("stats_bias/"+k, np.mean(v)) for k,v in self.bias_stats.items()]
        # logs += [("stats_bias/abs/"+k, np.mean(np.abs(v))) for k,v in self.bias_stats.items()]
        # self.bias_stats = defaultdict(list)

        # return logs

    def _random_action(self, n):
        if self.action_type == "continuous":
            return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))
        elif self.action_type == "discrete":
            # already reduced to 1 dim
            return np.random.choice(self.action_space.n, n).reshape(-1, 1)
        else:
            raise ValueError("The action type is not supported")
    
    def _preprocess_og(self, o, ag, g, ):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return actions, None, None, None

    def action_only(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.target  #self.target if use_target_net else
        action = self.sess.run(policy.pi_tf, feed_dict={
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        })
        return action
    

    def get_action_and_Q_pi(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.target  #self.target if use_target_net else
        action, q_pi = self.sess.run([policy.pi_tf, policy.Q_pi_tf], feed_dict={
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        })
        return action, q_pi

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        if noise_eps == 0:
            vals = [policy.deter_pi_tf]
        else:
            vals = [policy.pi_tf]
            
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        
        
        u = ret[0]
        if self.action_type == "continuous":
            noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.max_u, self.max_u)
            u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
            if u.shape[0] == 1:
                u = u[0]
            u = u.copy()
        elif self.action_type == "discrete":
            # no noise eps
            # random_eps is quite high
            u = np.argmax(u, axis=1).reshape(-1, 1)
            u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        # if self.action_type == "discrete":
        #     # discrete has different strategy of epsilon greedy.
        #     ret = 

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
    

    # actually not used
    def get_Q(self, o, g, u):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.target  # main
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu)
        }
        ret = self.sess.run(policy.Q_tf, feed_dict=feed)
        return ret

    def get_Q_pi(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        policy = self.target
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf:g.reshape(-1, self.dimg)
        }
        # I have taken the minimum of two critic networks
        ret = self.sess.run(policy.Q_pi_tf, feed_dict=feed)
        return ret

    # def get_target_Q(self, o, g, a, ag):
    #     o, g = self._preprocess_og(o, ag, g)
    #     policy = self.target
    #     # feed
    #     feed = {
    #         policy.o_tf: o.reshape(-1, self.dimo),
    #         policy.g_tf: g.reshape(-1, self.dimg),
    #         policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32) #??
    #     }

    #     ret = self.sess.run(policy.Q_tf, feed_dict=feed)
    #     return ret
    
    def inflat_actions(self, actions):
        if self.action_type == "discrete":
            shape = actions.shape
            num_actions = self.action_space.n
            actions = np.eye(num_actions)[actions.reshape(-1)]
            actions = actions.reshape(*shape[:-1], actions.shape[-1])
        
        return actions

    def store_episode(self, episode_batch, update_stats=True): #init=False
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key 'o' is of size T+1, others are of size T
        """
        episode_batch['u'] = self.inflat_actions(episode_batch['u'])
        self.buffer.store_episode(episode_batch)
        if update_stats:
            # episode doesn't has key o_2
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            # add transitions to normalizer
            transitions = self.samplers['her'](episode_batch, num_normalizing_transitions, self.her_info)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats
            # training normalizer online 
            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            # if self.use_dynamic_nstep:
            self.u_stats.update(transitions['u'])
            self.u_stats.recompute_stats()
    
    def get_reg_loss(self):
        logs = [('stats_loss/mean_' + k, np.mean(v)) for k,v in self.loss_stats.items()]
        self.loss_stats = defaultdict(list)
        return logs
    
    def get_mix_stats(self):
        logs = [('mix/mean_' + k, np.mean(v)) for k,v in self.mix_stats.items()]
        self.mix_stats = defaultdict(list)
        return logs

    def get_grad_norm(self):
        logs = [('grad_norm/mean_' + k, np.mean(v)) for k,v in self.grad_stats.items()]
        self.grad_stats = defaultdict(list)
        return logs

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad, \
            Q_grad_norm, pi_grad_norm, mix_ratio, mix_target, q_diff, exp_q_diff = self.sess.run([
            # self.one_step_bias_tf,
            self.Q_loss_tf,
            self.pi_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
            self.Q_grad_norm,
            self.pi_grad_norm,
            self.mix_ratio,
            self.mix_target,
            self.q_diff,
            self.exp_q_diff
        ])
        # self.bias_stats["ft_one_step_bias"].append(one_step_bias)

        self.loss_stats["critic_loss"].append(critic_loss)
        self.grad_stats["q_grad_norm"].append(Q_grad_norm)
        self.grad_stats["pi_grad_norm"].append(pi_grad_norm)
        self.mix_stats["mix_ratio"].append(mix_ratio)
        self.mix_stats["mix_target"].append(mix_target)

        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def update_dynamic_model(self, init=False):
        times = 1
        if init:
            times = self.dynamic_init
        for _ in range(times):
            transitions = self.buffer.sample(self.dynamic_batchsize)
            loss = self.dynamic_model.update(transitions['o'], transitions['u'], transitions['o_2'])

    def sample_batch(self, method='list'):
        transitions = self.buffer.sample(self.batch_size)   #otherwise only sample from primary buffer
        # o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        # ag, ag_2 = transitions['ag'], transitions['ag_2']
        # transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        # transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        if method == 'list':
            transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        else:
            transitions_batch = transitions
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss
    
    def update_info(self):
        if self.mode == "nstep_backward_gcdp":
            self.info["offset_limit"] += 1
        self.buffer.update_info(self.info)

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)


    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('u_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.u_stats = Normalizer(self.dimu, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        batch_tf['success_within_n_steps'] = tf.reshape(batch_tf['success_within_n_steps'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] , *clip_range)  # lambda target 
        self.target_tf = target_tf

        q1_to_goal = self.main.Q1_tf
        q2_to_goal = self.main.Q2_tf
        q_to_goal = tf.minimum(q1_to_goal, q2_to_goal)
        q_pi_to_goal = self.main.Q1_pi_tf
        pi_to_goal = self.main.pi_tf

        q_diff = tf.stop_gradient(target_tf) - q_to_goal
        self.q_diff = q_diff
        if self.clip_power:
            self.exp_q_diff = tf.math.exp(tf.stop_gradient(tf.clip_by_value(tf.pow(q_diff, self.exp_degree), -self.clip_exp, self.clip_exp)/self.temperature))
        else:
            self.exp_q_diff = tf.clip_by_value(tf.math.exp(tf.stop_gradient(tf.pow(q_diff, self.exp_degree)/self.temperature)), -self.clip_exp, self.clip_exp)

        
        self.mix_ratio = 1/(1 + self.exp_q_diff)
        self.mix_target = (tf.stop_gradient(target_tf) * self.exp_q_diff + tf.stop_gradient(q_to_goal)) * self.mix_ratio

        success_scale = batch_tf['success_within_n_steps'] * (self.success_scale - 1) + 1
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(self.mix_target) - q1_to_goal) * success_scale) +  tf.reduce_mean(tf.square(tf.stop_gradient(self.mix_target) - q2_to_goal) * success_scale)

        self.pi_loss_tf = -tf.reduce_mean(q_pi_to_goal)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(pi_to_goal/ self.max_u))

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        self.Q_grad_norm = tf.math.reduce_mean([tf.norm(grad) for grad in Q_grads_tf])
        self.pi_grad_norm = tf.math.reduce_mean([tf.norm(grad) for grad in pi_grads_tf])

        if self.grad_clip_value > 0:
            grad_clip_value = self.grad_clip_value
            Q_grads_tf = [tf.clip_by_value(grad, -grad_clip_value, grad_clip_value) for grad in Q_grads_tf]
            pi_grads_tf = [tf.clip_by_value(grad, -grad_clip_value, grad_clip_value) for grad in pi_grads_tf]


        # TODO add a parameter grad_clip_value
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # create the dynamic model wahtever.
        self.dynamic_model = ForwardDynamicsNumpy(self.dimo, self.dimu)
        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')

        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        logs += [('stats_u/mean', np.mean(self.sess.run([self.u_stats.mean])))]
        logs += [('stats_u/std', np.mean(self.sess.run([self.u_stats.std])))]
        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)


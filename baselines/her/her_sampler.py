import numpy as np
import gym
import gym_simple_minigrid
import multiworld
from scipy.stats import norm
import time

norm_dist = norm(0, 1)
interval = np.linspace(-4, 4, 81)
expect_look_up_table = np.array([norm_dist.expect(lambda x: np.minimum(x, y)) for y in interval])
def look_up_expect(mu, std, v):
    norm_v = (v - mu) / std
    norm_v = np.minimum(4, norm_v)
    norm_expect = np.zeros_like(norm_v)
    almost_no_change_indices = (norm_v < -4)
    look_up_indices = (1 - almost_no_change_indices).astype(np.bool)
    norm_expect[almost_no_change_indices] = norm_v[almost_no_change_indices]
    norm_expect[look_up_indices] = expect_look_up_table[np.round((norm_v[look_up_indices] + 4)/0.1).astype(np.int32)]
    expect = norm_expect * std + mu
    return expect





def make_random_sample(reward_fun):
    def _random_sample(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # # Re-compute reward since we may have substituted the u and o_2 ag_2
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
    return _random_sample
        
def obs_to_goal_fun(env):
    # only support Fetchenv and Handenv now
    from gym.envs.robotics import FetchEnv, hand_env
    from multiworld.envs.pygame import point2d
    from multiworld.envs.mujoco.sawyer_xyz import sawyer_push_and_reach_env
    from multiworld.envs.mujoco.sawyer_xyz import sawyer_reach

    if hasattr(env, 'env'):
        if isinstance(env.env, FetchEnv):
            obs_dim = env.observation_space['observation'].shape[0]
            goal_dim = env.observation_space['desired_goal'].shape[0]
            temp_dim = env.sim.data.get_site_xpos('robot0:grip').shape[0]
            def obs_to_goal(observation):
                observation = observation.reshape(-1, obs_dim)
                if env.has_object:
                    goal = observation[:, temp_dim:temp_dim + goal_dim]
                else:
                    goal = observation[:, :goal_dim]
                return goal.copy()
        elif isinstance(env.env, hand_env.HandEnv):
            goal_dim = env.observation_space['desired_goal'].shape[0]
            def obs_to_goal(observation):
                goal = observation[:, -goal_dim:]
                return goal.copy()
        elif isinstance(env.env, point2d.Point2DEnv):
            def obs_to_goal(observation):
                return observation.copy()
        elif isinstance(env.env, sawyer_push_and_reach_env.SawyerPushAndReachXYZEnv):
            assert env.env.observation_space['observation'].shape == env.env.observation_space['achieved_goal'].shape, \
                "This environment's observation space doesn't equal goal space"
            def obs_to_goal(observation):
                return observation
        elif isinstance(env.env, sawyer_reach.SawyerReachXYZEnv):
            def obs_to_goal(observation):
                return observation
        else:
            def obs_to_goal(observation):
                return observation
            # raise NotImplementedError('Do not support such type {}'.format(env))
    elif isinstance(env, gym_simple_minigrid.minigrid.SimpleMiniGridEnv):
            def obs_to_goal(observation):
                return observation[:, :2]
    else:
        raise ValueError("The env is not supported for MMHER")

        
    return obs_to_goal


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, obs_to_goal_fun=None, no_her=False):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0
    
    if no_her:
        print( '*' * 10 + 'Do not use HER in this method' + '*' * 10)
    
    def _random_log(string):
        if np.random.random() < 0.002:
            print(string)
    
    def _preprocess(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        # Select which episodes and time steps to use. 
        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_reward(ag_2, g):
        # Reconstruct info dictionary for reward  computation.
        info = {}
        # for key, value in transitions.items():
        #     if key.startswith('info_'):
        #         info[key.replace('info_', '')] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        return reward_fun(**reward_params)

    def _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, offset=False, offset_limit=np.inf):
        # future strategy
        her_indexes = (np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * np.minimum((T - t_samples), offset_limit)
        future_offset = future_offset.astype(int)
        # reset the offset to be as large as the horizon.
        no_her_indexes = (1-her_indexes).astype(np.bool)
        future_offset[no_her_indexes] = T - t_samples[no_her_indexes] - 1 
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        if offset:
            return future_ag.copy(), her_indexes.copy(), future_offset.copy()
        return future_ag.copy(), her_indexes.copy()

    def _reshape_transitions(transitions, batch_size, batch_size_in_transitions):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        gamma, Q_pi_fun, Q_fun = info['gamma'], info['get_Q_pi'], info['get_Q']
        # get transition in the vanilla way.
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        
        if not no_her:
            future_ag, her_indexes, future_offset = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, offset=True)
            transitions['g'][her_indexes] = future_ag


        reward =  _get_reward(transitions['ag_2'], transitions['g'])

        success = (reward == 0)
        Q_heads = Q_pi_fun(o=transitions['o_2'], g=transitions['g'], success=success)
        Q_largest_quantile = Q_heads[:, 0, None]

        if info.get('et', False):
            transitions['target_q'] = reward + gamma *  Q_largest_quantile.reshape(-1) * (1-success)
        else:
            transitions['target_q'] = reward + gamma *  Q_largest_quantile.reshape(-1)

        transitions['reward'] = reward
        transitions['success_within_n_steps'] = transitions['success_within_one_step'] = success.astype(np.float32)
        transitions['truncated_n_steps'] = np.ones(batch_size)

        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    def _sample_nstep_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_pi_fun, truncate = info['nstep'], info['gamma'], info['get_Q_pi'], info['truncate']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        _random_log('using nstep truncated sampler with step:{}'.format(steps))

        assert steps < T, 'Steps should be much less than T.'
        # I will consider using rewards instead of distance.

        n_step_ags = np.zeros((batch_size, steps, episode_batch['ag'].shape[-1]))
        # the mask will benefit the accumulation of rewards.
        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])
        for i in range(steps):
            i_t_samples = t_samples + i
            n_step_reward_mask[:,i][np.where(i_t_samples > T - 1)] = 0 # this only marks exceeding the horizon.
            i_t_samples[i_t_samples > T-1] = T-1   # last state to compute reward
            n_step_ags[:,i,:] = episode_batch['ag_2'][episode_idxs, i_t_samples] # ag_2 has involved the offset one.

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            future_ag, her_indexes, future_offset = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, offset=True)
            transitions['g'][her_indexes] = future_ag
        
        n_step_gs = transitions['g'].repeat(steps, axis=0)
        # Re-compute reward since we may have substituted the goal.
        ags = n_step_ags.reshape((batch_size * steps, -1))
        n_step_reward = _get_reward(ags, n_step_gs)
        n_step_reward = n_step_reward.reshape((batch_size, steps))
        # assuming the success is euqal to reward = 0
        n_step_success = (n_step_reward == 0)
        success_within_n_steps = np.max(n_step_success, axis=1)

        if truncate:
            terminal_mask = np.cumprod(1 - n_step_success, axis=1)
        else:
            terminal_mask = np.ones_like(n_step_success)

        non_terminal_ones = np.sum(terminal_mask, axis=1) # 0: 1, 1: 2, 2: 2
        truncated_n_steps = np.clip(non_terminal_ones + 1, 1, steps)

        i_t_samples = t_samples + truncated_n_steps # last state to observe
        i_t_samples[i_t_samples > T] = T
        n_step_os = episode_batch['o'][episode_idxs, i_t_samples] # the next observation.
        
        true_offset = i_t_samples - t_samples
        truncated_n_steps = np.minimum(true_offset, truncated_n_steps)
        assert np.min(truncated_n_steps) >= 1, "the truncated n step is incorrect"

        n_step_gamma = pow(gamma, truncated_n_steps).reshape(-1, 1)
        transitions['n_step_cumulative_rewards'] = (n_step_reward * n_step_reward_mask * terminal_mask).sum(axis=1).copy()
        transitions['success_within_n_steps'] = success_within_n_steps.astype(np.float32)
        transitions['success_within_one_step'] = n_step_success[:, 0].astype(np.float32)
        transitions['truncated_n_steps'] = truncated_n_steps
        # n_step_os is incorrect now.

        Q_heads = Q_pi_fun(o=n_step_os, g=transitions['g'], success=success_within_n_steps)
        Q_largest_quantile = Q_heads[:, 0, None]

        if info.get('et', False):
            transitions['target_q'] = transitions['n_step_cumulative_rewards'] + (n_step_gamma * Q_largest_quantile).reshape(-1) * (1-success_within_n_steps)
        else:
            transitions['target_q'] = transitions['n_step_cumulative_rewards']  + (n_step_gamma * Q_largest_quantile).reshape(-1)
        if np.random.random() < 0.001:
            print(np.mean(transitions['target_q']))
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    def _sample_nstep_lambda_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_pi_fun, lamb, truncate = info['nstep'], info['gamma'], info['get_Q_pi'], info['lamb'], info['truncate']
        transitions, episode_idxs, t_samples, batch_size, T= _preprocess(episode_batch, batch_size_in_transitions)
        assert steps < T, 'Steps should be much less than T.'

        _random_log('using nstep lambda sampler with step:{} and lamb:{}'.format(steps, lamb))

        n_step_ags = np.zeros((batch_size, steps, episode_batch['ag'].shape[-1]))
        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])
        n_step_o2s= np.zeros((batch_size, steps, episode_batch['o'].shape[-1]))
        n_step_gamma_matrix = np.ones((batch_size, steps))  # for lambda * Q

        for i in range(steps):
            i_t_samples = t_samples + i
            n_step_reward_mask[:,i][np.where(i_t_samples > T - 1)] = 0
            n_step_gamma_matrix[:,i] = pow(gamma, i+1)
            if i >= 1:  # more than length, use the last one
                n_step_gamma_matrix[:,i][np.where(i_t_samples > T -1)] = n_step_gamma_matrix[:, i-1][np.where(i_t_samples > T-1)]
            i_t_samples[i_t_samples > T-1] = T-1   # last state to compute reward
            n_step_ags[:,i,:] = episode_batch['ag_2'][episode_idxs, i_t_samples]
            n_step_o2s[:,i,:] = episode_batch['o_2'][episode_idxs, i_t_samples]

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            future_ag, her_indexes, future_offset = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, offset=True)
            transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        n_step_gs = transitions['g'].repeat(steps, axis=0)
        ags = n_step_ags.reshape((batch_size * steps, -1))
        n_step_reward = _get_reward(ags, n_step_gs).reshape((batch_size, steps))
        n_step_success = (n_step_reward == 0)
        success_within_n_steps = np.max(n_step_success, axis=1)

        if truncate:
            terminal_mask = np.cumprod(1 - n_step_success, axis=1)
        else:
            terminal_mask = np.ones_like(n_step_success)

        non_terminal_ones = np.sum(terminal_mask, axis=1) # 0: 1, 1: 2, 2: 2
        truncated_n_steps = np.clip(non_terminal_ones + 1, 1, steps)

        i_t_samples = t_samples + truncated_n_steps # last state to observe
        i_t_samples[i_t_samples > T] = T
        true_offset = i_t_samples - t_samples
        truncated_n_steps = np.minimum(true_offset, truncated_n_steps)
        assert np.min(truncated_n_steps) >= 1, "the truncated n step is incorrect"

        transitions['success_within_n_steps'] = success_within_n_steps.astype(np.float32)
        transitions['success_within_one_step'] = n_step_success[:, 0].astype(np.float32)
        transitions['truncated_n_steps'] = truncated_n_steps

        return_array = np.zeros((batch_size, steps))
        return_mask = np.ones((batch_size, steps)) * np.array([pow(lamb,i) for i in range(steps)]) # interpolate the return over different steps.

        n_step_gamma = pow(gamma, truncated_n_steps).reshape(-1, 1)


        for i in range(steps):
            i_t = np.minimum(i, truncated_n_steps - 1)
            i_step_gamma = np.maximum(n_step_gamma, n_step_gamma_matrix[:, i].reshape(-1, 1)).reshape(-1)
            i_step_o2s = n_step_o2s[np.arange(batch_size), i_t].reshape((batch_size, episode_batch['o'].shape[-1]))
            i_step_success = n_step_success[np.arange(batch_size), i_t].astype(np.float32)
            Q_heads = Q_pi_fun(o=i_step_o2s, g=transitions['g'], success=i_step_success)
            Q_largest_quantile = Q_heads[:, 0, None].reshape(-1)
            if info.get('et', False):

                success_within_i_steps = np.max(n_step_success[:i+1], axis=1)

                return_i = (n_step_reward[:,:i+1] * n_step_reward_mask[:,:i+1] * terminal_mask[:, :i+1]).sum(axis=1) + \
                    i_step_gamma * Q_largest_quantile * (1 - success_within_i_steps)
            else:
                return_i = (n_step_reward[:,:i+1] * n_step_reward_mask[:,:i+1] * terminal_mask[:, :i+1]).sum(axis=1) + \
                    i_step_gamma * Q_largest_quantile
                    
            return_array[:, i] = return_i.copy()
        transitions['target_q'] = ((return_array * return_mask).sum(axis=1) / return_mask.sum(axis=1)).copy()
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    def _sample_nstep_dynamic_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_pi_fun, Q_fun, alpha, truncate = info['nstep'], info['gamma'], info['get_Q_pi'], info['get_Q'], info['alpha'], info['truncate']
        dynamic_model, action_and_Q_pi_fun = info['dynamic_model'], info['get_action_and_Q_pi']
        transitions, episode_idxs, t_samples, batch_size, T= _preprocess(episode_batch, batch_size_in_transitions)

        _random_log('using nstep dynamic sampler with step:{} and alpha:{}'.format(steps, alpha))

        # preupdate dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions['o_2'], times=2)
        # print(loss)

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            future_ag, her_indexes, future_offset = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, offset=True)
            transitions['g'][her_indexes] = future_ag

        # transitions = _get_anchor_information(transitions, episode_batch, episode_idxs, batch_size, t_samples, her_indexes, future_offset, anchor_offset_limit)

        # Re-compute reward since we may have substituted the goal.
        ## model-based on-policy
        reward_list = [_get_reward(transitions['ag_2'], transitions['g'])]
        last_state = transitions['o_2']
        Q_pi_cache = []
        success= (transitions['r'] == 0).astype(np.float32)
        if steps > 1:
            for _ in range(1, steps):
                state_array = last_state
                # action_fun is the policy function.
                action_array, Q_pi = action_and_Q_pi_fun(o=state_array, g=transitions['g'], success=success)
                next_state_array = dynamic_model.predict_next_state(state_array, action_array)
                # test loss
                predicted_obs = dynamic_model.predict_next_state(state_array, transitions['u'])
                # should be the next transition.
                loss = np.abs((transitions['o_2'] - predicted_obs)).mean()
                if np.random.random() < 0.001:
                    print(loss)
                    # print(transitions['o_2'][0])
                    # print(predicted_obs[0])
                
                Q_pi_cache.append(Q_pi[:, 0, None].reshape(-1))
               
                next_reward = _get_reward(obs_to_goal_fun(next_state_array), transitions['g'])
                reward_list.append(next_reward.copy())
                success = (next_reward == 0).astype(np.float32)
                last_state = next_state_array
        Q_pi_cache.append(Q_pi_fun(o=last_state, g=transitions['g'], success=success).reshape(-1))
        Q_pi_cache = np.stack(Q_pi_cache, axis=1)
        transitions['reward'] = reward_list[0]

        n_step_reward = np.stack(reward_list, axis=1)
        n_step_success = (n_step_reward == 0)
        success_within_n_steps = np.max(n_step_success, axis=1)

        if truncate:
            terminal_mask = np.cumprod(1 - n_step_success, axis=1)
        else:
            terminal_mask = np.ones_like(n_step_success)

        non_terminal_ones = np.sum(terminal_mask, axis=1) # 0: 1, 1: 2, 2: 2
        truncated_n_steps = np.clip(non_terminal_ones + 1, 1, steps)

        i_t_samples = t_samples + truncated_n_steps # last state to observe
        i_t_samples[i_t_samples > T] = T
        true_offset = i_t_samples - t_samples
        truncated_n_steps = np.minimum(true_offset, truncated_n_steps)
        assert np.min(truncated_n_steps) >= 1, "the truncated n step is incorrect"

        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])

        transitions['target_q'] = (n_step_reward * n_step_reward_mask * terminal_mask).sum(axis=1).copy()
        transitions['success_within_n_steps'] = success_within_n_steps.astype(np.float32)
        transitions['success_within_one_step'] = n_step_success[:, 0].astype(np.float32)
        transitions['truncated_n_steps'] = truncated_n_steps


        if info.get('et', False):
            transitions['target_q'] += pow(gamma, truncated_n_steps) * Q_pi_cache[np.arange(batch_size), truncated_n_steps-1].reshape(-1) * (1-success_within_n_steps)
            target_step1 = reward_list[0] + gamma * Q_pi_cache[:, 0] * (1-n_step_success[:, 0])
        else:
            transitions['target_q'] += pow(gamma, truncated_n_steps) * Q_pi_cache[np.arange(batch_size), truncated_n_steps-1].reshape(-1)
            # allievate the model bias
            target_step1 = reward_list[0] + gamma * Q_pi_cache[:, 0]
        
        transitions['one_step_r'] = target_step1
        if steps > 1:
            transitions['target_q'] = (alpha * transitions['target_q'] + target_step1) / (1 + alpha)
        
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)
    
    samplers = {
        'her': _sample_her_transitions,
        'nstep':_sample_nstep_her_transitions,
        'lambda':_sample_nstep_lambda_her_transitions,
        'dynamic':_sample_nstep_dynamic_her_transitions,
    }

    return samplers


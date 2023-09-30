import os

import click
import numpy as np
from mpi4py import MPI
import time
from copy import copy

from baselines import logger
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import dump_params

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

def log_successful_trajectory_bias(bias_stat, success_bias_dict, success_array):
    if np.sum(success_array) != 0:
        for k, v in success_bias_dict.items():
            bias_stat["bias/success_" + k] = np.sum(v * success_array) / np.sum(success_array)
            bias_stat["bias/success_abs_" + k] = np.sum(np.abs(v) * success_array) / np.sum(success_array)
    else:
        for k, v in success_bias_dict.items():
            bias_stat["bias/success_" + k] = -1
            bias_stat["bias/success_abs_" + k] = -1

    return bias_stat

def log_failed_trajecotry_bias(bias_stat, failure_bias_dict, success_array):
    if np.min(success_array) == 0:
        for k, v in failure_bias_dict.items():
            bias_stat["bias/failure_" + k] = np.sum(v * (1 - success_array))/np.sum(1 - success_array)
            bias_stat["bias/failure_abs_" + k] = np.sum(np.abs(v) * (1 - success_array))/np.sum(1 - success_array)
    else:
        for k, v in failure_bias_dict.items():
            bias_stat["bias/failure_" + k] = -1
            bias_stat["bias/failure_abs_" + k] = -1
    return bias_stat
    

def evaluate(evaluator, logger, n_test_rollouts):
    evaluator.clear_history()
    evaluator.render = True
    bias_stat = {}
    istb_bias_list = []
    success_list = []
    shifting_bias_list = []
    initial_shooting_bias_list = []
    average_shooting_bias_list = []

    for _ in range(n_test_rollouts):
        _, istb_bias, shifting_bias, initial_shooting_bias, average_shooting_bias, \
            success = evaluator.generate_rollouts()
        istb_bias_list.append(istb_bias)
        shifting_bias_list.append(shifting_bias)
        initial_shooting_bias_list.append(initial_shooting_bias)
        average_shooting_bias_list.append(average_shooting_bias)
        success_list.append(success)

    istb_bias_array = np.concatenate(istb_bias_list)
    shifting_bias_array = np.concatenate(shifting_bias_list)
    initial_shooting_bias_array = np.concatenate(initial_shooting_bias_list)
    average_shooting_bias_array = np.concatenate(average_shooting_bias_list)
    success_array = np.concatenate(success_list)

    bias_stat["mean_abs_bias"] = np.mean(np.abs(istb_bias_array))
    bias_stat["average_bias"] = np.mean(istb_bias_array)

    bias_data = {"initial": istb_bias_array, "shifting": shifting_bias_array, "initial_shooting": initial_shooting_bias_array,
                "average_shooting": average_shooting_bias_array}

    bias_stat = log_successful_trajectory_bias(bias_stat, bias_data, success_array)
    bias_stat = log_failed_trajecotry_bias(bias_stat, bias_data, success_array)

    for key, val in bias_stat.items():
            logger.record_tabular(key, mpi_average(val))

def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, n_eps_per_cycle, policy_save_interval,
          save_path, demo_file, random_init, debug_path, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    
    # random_init buffer and o/g/rnd stat 
    if random_init:
        logger.info('Random initializing ...')
        rollout_worker.clear_history()
        for epi in range(int(random_init) // rollout_worker.rollout_batch_size): 
            episode = rollout_worker.generate_rollouts(random_ac=True)
            policy.store_episode(episode)
        if policy.mode == 'dynamic' and policy.n_step > 1:
            policy.update_dynamic_model(init=True)

    best_success_rate = -1
    logger.info('Start training...')
    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    n_rounds = n_eps_per_cycle//rollout_worker.rollout_batch_size
    remain = n_eps_per_cycle%rollout_worker.rollout_batch_size
    info_to_dump = {}
    for epoch in range(n_epochs):
        # from baselines.her.util import write_to_file
        # write_to_file('\n epoch: {}'.format(epoch))
        time_start = time.time()
        # train
        policy.set_epoch(epoch, save_path)
        rollout_worker.clear_history()
        for i in range(n_cycles):
            if remain != 0:
                print("WARNING: the actual episode for each batch is", n_rounds * rollout_worker.rollout_batch_size)
            for _ in range(n_rounds):
                policy.dynamic_batch = False
                episode = rollout_worker.generate_rollouts()
                policy.store_episode(episode)
            for j in range(n_batches):   
                policy.train()
            # policy.update_target_net()
        policy.update_info()
        
        # test
        evaluator.clear_history()

        if debug_path is not None:
            tf_util.load_variables(debug_path)
            print("load the model from path %s succesfully" % debug_path)

        # record logs
        time_end = time.time()
        logger.record_tabular('epoch', epoch)
        logger.record_tabular('epoch time(min)', (time_end - time_start)/60)

        evaluate(evaluator, logger, n_test_rollouts)


        for key, val in policy.get_loss_stat():
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.get_debug_info():
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.get_grad_norm():
            logger.record_tabular(key, mpi_average(val))

        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate > best_success_rate and save_path:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy

def learn(*, network, env, num_epoch, total_timesteps,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    random_init=0,
    play_no_training=False,
    debug_path=None,
    **kwargs
):

    override_params = override_params or {} 
    rank = MPI.COMM_WORLD.Get_rank()
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.spec.id

    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter

    params.update(kwargs)   # make kwargs part of params
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs
    random_init = params['random_init']
    # save total params
    dump_params(logger, params)

    if rank == 0:
        config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***') 
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(action_space=copy(env.action_space), dims=dims, params=params, clip_return=clip_return)
    if load_path is not None:
        tf_util.load_variables(load_path)
    
    # no training
    if play_no_training:  
        return policy

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        'debug_dir': "/".join(debug_path.split('/')[:-1]) if debug_path is not None else None
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env

    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    n_cycles = params['n_cycles']
    if num_epoch:  # prefer to use num_poch
        n_epochs = num_epoch
    else:
        ## epochs = total_steps // n_cycles // (rollout_worker.steps * roolout_worker.episodes), often less than we think
        n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size

    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'], n_eps_per_cycle=12,
        policy_save_interval=policy_save_interval, demo_file=demo_file, random_init=random_init,
        debug_path=debug_path)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--num_epoch', type=int, default = '100', help='number of epochs to train')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()

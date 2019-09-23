import sys
import os
from shutil import rmtree
from copy import deepcopy
from multiprocessing import Manager
from tensorboardX import SummaryWriter
import logging
import json
from tempfile import mkdtemp
import numpy as np
from git import Repo, InvalidGitRepositoryError

logger = logging.getLogger('Configurator')

from task.task import Task
from task_selector.task_selector import TaskSelector
from task_planner.task_planner import TaskPlanner
from rollout.parallel_rollout_manager import ParallelRolloutManager
from parallel_train_manager import ParallelTrainManager
from forward_model.forward_model import ForwardModel
from utils import get_parameter, update_default_params, import_function, create_seed
from rollout.rollout_worker import RolloutWorker
from gnet.gnet import Gnet

def basic_configure(**kwargs):
    ##########################
    #     load experiment    #
    ##########################

    experiments_path = []
    if 'experiment' in kwargs and kwargs['experiment'] is not None:
        experiments_kwargs = []
        experiments_path = [os.path.splitext(os.path.basename(kwargs['experiment']))[0]]
        experiment_basedir = os.path.dirname(kwargs['experiment'])
        while True:
            with open(os.path.join(experiment_basedir, experiments_path[-1] + '.json'), 'r') as f:
                experiments_kwargs.append(json.load(f))
            if experiments_kwargs[-1]['inherit_from'] is not None:
                experiments_path.append(experiments_kwargs[-1]['inherit_from'])
                continue
            break
        for experiment_kwargs in reversed(experiments_kwargs):
            update_default_params(kwargs, experiment_kwargs)

    ##########################
    #     load json string   #
    ##########################

    if 'json_string' in kwargs and kwargs['json_string'] is not None:
        update_default_params(kwargs, json.loads(kwargs['json_string']))

    ##########################
    #     Prepare logging    #
    ##########################

    clean = get_parameter('clean', params=kwargs, default=False)
    jobdir = get_parameter('jobdir', params=kwargs, default=mkdtemp())

    if clean and os.path.exists(jobdir) and not os.path.exists(os.path.join(jobdir, 'restart')):
        rmtree(jobdir)

    os.makedirs(jobdir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] <%(levelname)s> %(name)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        handlers=([
                                   logging.FileHandler(os.path.join(jobdir, 'events.log'))] +
                                  [logging.StreamHandler(sys.stdout)])
                        )

    summary_writer = SummaryWriter(os.path.join(jobdir, 'tf_board'))

    if clean: logger.info(f'Cleaned jobdir {jobdir}')

    for experiment_path in reversed(experiments_path):
        logger.info(f'Loaded params from experiment {experiment_path}')

    project_path = os.path.dirname(os.path.realpath(__file__))
    try:
        repo = Repo(project_path, search_parent_directories=True)
        active_branch = repo.active_branch
        latest_commit = repo.commit(active_branch)
        latest_commit_sha = latest_commit.hexsha
        latest_commit_sha_short = repo.git.rev_parse(latest_commit_sha, short=6)
        logger.info(f'We are on branch {active_branch} using commit {latest_commit_sha_short}')
    except InvalidGitRepositoryError:
        logger.warn(f'{project_path} is not a git repo')

    ##########################
    #    Continue training   #
    ##########################

    restart_after = get_parameter('restart_after', params=kwargs, default=None)
    continued_params = {}
    if restart_after is not None and os.path.exists(os.path.join(jobdir, 'restart')):
        with open(os.path.join(jobdir, 'basic_params.json'), 'r') as f:
            continued_params = json.load(f)

    ##########################
    #  Load external config  #
    ##########################

    basic_params_path = get_parameter('basic_params_path', params=continued_params, default=None)
    basic_params_path = get_parameter('basic_params_path', params=kwargs, default=basic_params_path)
    external_params = {}
    if basic_params_path is not None:
        with open(basic_params_path, 'r') as f:
            external_params = json.load(f)

    ##########################
    #        Seeding         #
    ##########################

    seed = get_parameter('seed', params=external_params, default=int(np.random.random_integers(0, 2**23-1)))
    seed = get_parameter('seed', params=continued_params, default=seed)
    seed = get_parameter('seed', params=kwargs, default=seed)

    logger.info(f'Using seed {seed}')

    ####################
    #    Prepare env   #
    ####################

    env_spec = get_parameter('env', params=external_params, default=None)
    env_spec = get_parameter('env', params=continued_params, default=env_spec)
    env_spec = get_parameter('env', params=kwargs, default=env_spec)

    env_params = dict()
    update_default_params(env_params, external_params.get('env_params', {}))
    update_default_params(env_params, continued_params.get('env_params', {}))
    update_default_params(env_params, kwargs.get('env_params', {}))

    env_proto = import_function(env_spec)
    tmp_env = env_proto(**env_params)
    obs = tmp_env.reset()
    env_spec = dict(
        o_dim=obs['observation'].shape[0],
        a_dim=tmp_env.action_space.shape[0],
        g_dim=obs['desired_goal'].shape[0],
    )
    if hasattr(tmp_env, 'goal_min'): env_spec['goal_min'] = tmp_env.goal_min
    if hasattr(tmp_env, 'goal_max'): env_spec['goal_max'] = tmp_env.goal_max
    update_default_params(env_spec, external_params.get('env_spec', {}))
    update_default_params(env_spec, continued_params.get('env_spec', {}))
    update_default_params(env_spec, kwargs.get('env_spec', {}))

    T = get_parameter('T', params=env_spec, default=800)


    env_fn = (env_proto, env_params)

    ####################
    #   Prepare tasks  #
    ####################

    tasks_specs = []
    update_default_params(tasks_specs, external_params.get('tasks_specs', {}))
    update_default_params(tasks_specs, continued_params.get('tasks_specs', {}))
    update_default_params(tasks_specs, kwargs.get('tasks_specs', {}))

    tasks_specs = [task_spec for task_spec in tasks_specs if task_spec.get('active', True)]

    tasks_fn = []
    for task_spec in tasks_specs:
        if 'active' in task_spec: del(task_spec['active'])
        task_spec['id'] = len(tasks_fn)
        task_spec['scope'] = f'Task{task_spec["id"]}'
        tasks_fn.append((Task, task_spec))

    ####################
    # Prepare policies #
    ####################

    policy_params = dict()
    update_default_params(policy_params, external_params.get('policy_params', {}))
    update_default_params(policy_params, continued_params.get('policy_params', {}))
    update_default_params(policy_params, kwargs.get('policy_params', {}))

    assert 'policy_type' in policy_params

    policy_proto = import_function(policy_params['policy_type'])

    policies_fn = []
    policies_params = []
    for task_spec in tasks_specs:
        params = deepcopy(policy_params)
        params['env_spec'] = env_spec
        params['task_spec'] = task_spec
        params['scope'] = f'policy_{task_spec["id"]}'
        del params['policy_type']
        policies_params.append(params)
        policies_fn.append((policy_proto, params))

    #########################
    # Prepare task selector #
    #########################

    task_selector_params = dict(
        tasks_specs=tasks_specs,
        surprise_weighting=0.1,
        buffer_size=100,
        lr=0.1,
        reg=1e-3,
        precision=1e-3,
        eps_greedy_prob=0.05,
        surprise_hist_weighting=.99,
        scope='taskSelector',
        fixed_Q=None,
        epsilon=0.1,
    )
    update_default_params(task_selector_params, external_params.get('task_selector_params', {}))
    update_default_params(task_selector_params, continued_params.get('task_selector_params', {}))
    update_default_params(task_selector_params, kwargs.get('task_selector_params', {}))

    task_selector_fn = (TaskSelector, task_selector_params)

    #########################
    # Prepare task planner  #
    #########################

    task_planner_params = dict(
        env_specs=env_spec,
        tasks_specs=tasks_specs,
        surprise_weighting=0.001,
        surprise_hist_weighting=.99,
        buffer_size=100,
        eps_greedy_prob=0.05,
        max_seq_length=10,
        scope='taskPlanner',
        fixed_Q=None,
        epsilon=0.0001,
    )
    update_default_params(task_planner_params, external_params.get('task_planner_params', {}))
    update_default_params(task_planner_params, continued_params.get('task_planner_params', {}))
    update_default_params(task_planner_params, kwargs.get('task_planner_params', {}))

    task_planner_fn = (TaskPlanner, task_planner_params)

    #########################
    #     Prepare gnets     #
    #########################

    gnet_params = dict(
        env_spec=env_spec,
        tasks_specs=tasks_specs,
        pos_buffer_size=int(1e3),
        neg_buffer_size=int(1e5),
        batch_size=64,
        learning_rate=1e-4,
        train_steps=100,
        only_fst_surprising_singal=True,
        only_pos_rollouts=False,
        normalize=False,
        normalizer_params=dict(
            eps=0.01,
            default_clip_range=5
        ),
        coords_gen_params=dict(
            buffer_size=int(1e5),
        ),
        reset_model_below_n_pos_samples=20,
        use_switching_reward=True,
    )
    update_default_params(gnet_params, external_params.get('gnet_params', {}))
    update_default_params(gnet_params, continued_params.get('gnet_params', {}))
    update_default_params(gnet_params, kwargs.get('gnet_params', {}))

    assert 'network_params' in gnet_params

    gnets_fn = []
    gnets_params = []
    for i in range(len(tasks_specs)):
        gnets_fn.append([])
        gnets_params.append([])
        for j in range(len(tasks_specs)):
            params = deepcopy(gnet_params)
            params['task_from_id'] = i
            params['task_to_id'] = j
            params['scope'] = f'gnet_{i}_to_{j}'
            gnets_params[-1].append(params)
            gnets_fn[-1].append((Gnet, params))

    #########################
    # Prepare forward model #
    #########################

    forward_model_params = dict(
        env_spec=env_spec,
        buffer_size=int(1e6),
        lr=1e-4,
        hist_length=1,
        batch_size=64,
        network_params=dict(
            nL=[100]*9,
            net_type='forward_model.models:ForwardModelMLPStateDiff',
            activation='tensorflow.nn:tanh',
            layer_norm=False,
            scope='mlp'
        ),
        normalizer_params=None,
        train_steps=100,
        scope='forwardModel'
    )
    update_default_params(forward_model_params, external_params.get('forward_model_params', {}))
    update_default_params(forward_model_params, continued_params.get('forward_model_params', {}))
    update_default_params(forward_model_params, kwargs.get('forward_model_params', {}))

    forward_model_fn = (ForwardModel, forward_model_params)

    #########################
    # Prepare RolloutWorker #
    #########################

    rollout_worker_params = dict(
        surprise_std_scaling=3,
        discard_modules_buffer=True,
        seed=seed,
        forward_model_burnin_eps=50,
        resample_goal_every=5,
    )
    update_default_params(rollout_worker_params, external_params.get('rollout_worker_params', {}))
    update_default_params(rollout_worker_params, continued_params.get('rollout_worker_params', {}))
    update_default_params(rollout_worker_params, kwargs.get('rollout_worker_params', {}))

    rollout_worker_fn = (RolloutWorker, rollout_worker_params)

    #########################
    # Write params to file  #
    #########################

    inherit_from = get_parameter('inherit_from', params=external_params, default=None)
    inherit_from = get_parameter('inherit_from', params=continued_params, default=inherit_from)
    inherit_from = get_parameter('inherit_from', params=kwargs, default=inherit_from)

    params_path = get_parameter('params_path', params=external_params, default=None)
    params_path = get_parameter('params_path', params=continued_params, default=params_path)
    params_path = get_parameter('params_path', params=kwargs, default=params_path)

    params_prefix = get_parameter('params_prefix', params=external_params, default=None)
    params_prefix = get_parameter('params_prefix', params=continued_params, default=params_prefix)
    params_prefix = get_parameter('params_prefix', params=kwargs, default=params_prefix)

    max_env_steps = get_parameter('max_env_steps', params=external_params, default=None)
    max_env_steps = get_parameter('max_env_steps', params=continued_params, default=max_env_steps)
    max_env_steps = get_parameter('max_env_steps', params=kwargs, default=max_env_steps)

    render = get_parameter('render', params=external_params, default=None)
    render = get_parameter('render', params=continued_params, default=render)
    render = get_parameter('render', params=kwargs, default=render)

    num_worker = get_parameter('num_worker', params=external_params, default=None)
    num_worker = get_parameter('num_worker', params=continued_params, default=num_worker)
    num_worker = get_parameter('num_worker', params=kwargs, default=num_worker)

    eval_runs = get_parameter('eval_runs', params=external_params, default=None)
    eval_runs = get_parameter('eval_runs', params=continued_params, default=eval_runs)
    eval_runs = get_parameter('eval_runs', params=kwargs, default=eval_runs)

    env = get_parameter('env', params=external_params, default=None)
    env = get_parameter('env', params=continued_params, default=env)
    env = get_parameter('env', params=kwargs, default=env)

    restart_after = get_parameter('restart_after', params=kwargs, default=restart_after)

    json_string = get_parameter('json_string', params=external_params, default=None)
    json_string = get_parameter('json_string', params=continued_params, default=json_string)
    json_string = get_parameter('json_string', params=kwargs, default=json_string)

    experiment = get_parameter('experiment', params=external_params, default=None)
    experiment = get_parameter('experiment', params=continued_params, default=experiment)
    experiment = get_parameter('experiment', params=kwargs, default=experiment)

    store_params_every = get_parameter('store_params_every', params=external_params, default=None)
    store_params_every = get_parameter('store_params_every', params=continued_params, default=store_params_every)
    store_params_every = get_parameter('store_params_every', params=kwargs, default=store_params_every)

    params_cache_size = get_parameter('params_cache_size', params=external_params, default=None)
    params_cache_size = get_parameter('params_cache_size', params=continued_params, default=params_cache_size)
    params_cache_size = get_parameter('params_cache_size', params=kwargs, default=params_cache_size)

    use_surprise = get_parameter('use_surprise', params=external_params, default=True)
    use_surprise = get_parameter('use_surprise', params=continued_params, default=use_surprise)
    use_surprise = get_parameter('use_surprise', params=kwargs, default=use_surprise)

    params = dict(
        inherit_from=inherit_from,
        basic_params_path=basic_params_path,
        params_path=params_path,
        params_prefix=params_prefix,
        store_params_every=store_params_every,
        params_cache_size=params_cache_size,
        use_surprise=use_surprise,
        seed=seed,
        clean=clean,
        jobdir=jobdir,
        max_env_steps=max_env_steps,
        render=render,
        num_worker=num_worker,
        eval_runs=eval_runs,
        env=env,
        restart_after=restart_after,
        json_string=json_string,
        experiment=experiment,
        env_spec=env_spec,
        env_params=env_params,
        tasks_specs=tasks_specs,
        policy_params=policy_params,
        policies_params=policies_params,
        task_selector_params=task_selector_params,
        task_planner_params=task_planner_params,
        gnet_params=gnet_params,
        gnets_params=gnets_params,
        forward_model_params=forward_model_params,
        rollout_worker_params=rollout_worker_params,
    )

    assert np.all([k in params for k in kwargs.keys()]), [k for k in kwargs.keys() if not k in params]

    with open(os.path.join(jobdir, 'basic_params.json'), 'w') as f:
        json.dump(params, f)

    return params, {'env_fn': env_fn, 'tasks_fn': tasks_fn, 'policies_fn': policies_fn,
            'gnets_fn': gnets_fn, 'task_selector_fn': task_selector_fn, 'task_planner_fn': task_planner_fn,
            'forward_model_fn': forward_model_fn, 'rollout_worker_fn': rollout_worker_fn}, summary_writer

def configure_parallel(**kwargs):

    basic_conf, modules_fn, summary_writer = basic_configure(**kwargs)

    jobdir = basic_conf['jobdir']
    env_spec = basic_conf['env_spec']
    T = env_spec['T']
    seed = basic_conf['seed']
    num_worker = basic_conf['num_worker']
    env_fn = modules_fn['env_fn']
    tasks_fn = modules_fn['tasks_fn']
    policies_fn = modules_fn['policies_fn']
    gnets_fn = modules_fn['gnets_fn']
    task_selector_fn = modules_fn['task_selector_fn']
    task_planner_fn = modules_fn['task_planner_fn']
    forward_model_fn = modules_fn['forward_model_fn']
    rollout_worker_fn = modules_fn['rollout_worker_fn']

    ##########################
    #    Continue training   #
    ##########################

    restart_after = get_parameter('restart_after', params=kwargs, default=None)
    continued_params = {}
    if restart_after is not None and os.path.exists(os.path.join(jobdir, 'restart')):
        with open(os.path.join(jobdir, 'parallel_params.json'), 'r') as f:
            continued_params = json.load(f)

    ##########################
    #  Load external config  #
    ##########################

    parallel_params_path = get_parameter('basic_params_path', params=continued_params, default=None)
    parallel_params_path = get_parameter('basic_params_path', params=kwargs, default=parallel_params_path)
    external_params = {}
    if parallel_params_path is not None:
        with open(parallel_params_path, 'r') as f:
            external_params = json.load(f)

    ###################################
    #  Prepare shared memory manager  #
    ###################################

    manager = Manager()
    episode = manager.dict()
    info = manager.dict()
    managed_memory = dict(
        episode=episode,
        info=info,
    )

    ##########################
    # Prepare rollout worker #
    ##########################

    parallel_rollout_manager_params = dict(
        num_worker=num_worker,
    )
    update_default_params(parallel_params_path, external_params.get('parallel_rollout_manager_params', {}))
    update_default_params(parallel_params_path, continued_params.get('parallel_rollout_manager_params', {}))
    update_default_params(parallel_rollout_manager_params, kwargs.get('parallel_rollout_manager_params', {}))

    parallel_rollout_manager = ParallelRolloutManager(env_spec, env_fn, tasks_fn, policies_fn, gnets_fn, task_selector_fn,
                                                      task_planner_fn, forward_model_fn, rollout_worker_fn, T,
                                                      managed_memory=managed_memory, **parallel_rollout_manager_params)

    ##########################
    #  Prepare train worker  #
    ##########################

    parallel_train_manager = ParallelTrainManager(managed_memory=managed_memory, summary_writer=summary_writer,
                                                  jobdir=jobdir, seed=seed)
    [parallel_train_manager.add_module(policy_fn) for policy_fn in policies_fn]
    [parallel_train_manager.add_module(task_fn) for task_fn in tasks_fn]
    parallel_train_manager.add_module(forward_model_fn)
    [parallel_train_manager.add_module(gnets_fn[i][j]) for i in range(len(tasks_fn)) for j in range(len(tasks_fn))]
    parallel_train_manager.add_module(task_selector_fn)
    parallel_train_manager.add_module(task_planner_fn)

    ##########################
    #  Load external params  #
    ##########################

    params_path = basic_conf['params_path']
    params_prefix = basic_conf['params_prefix']

    if params_path:
        params_path = params_path if params_path else jobdir
        try:
            parallel_train_manager.load_global_params(path=params_path, prefix=params_prefix)
            logger.info(f'Restored params from {params_path} with prefix {params_prefix}')
        except:
            logger.warning('Could not restore params')
            raise

    ##########################
    #    Continue training   #
    ##########################

    if basic_conf['restart_after'] is not None and os.path.exists(os.path.join(jobdir, 'restart')):
        try:
            parallel_train_manager.load_global_params(path=jobdir, prefix='latest')
            parallel_train_manager.restore_buffer(path=jobdir, prefix='latest')
            logger.info(f'Restored params from {params_path} with prefix {params_prefix}')
        except:
            logger.warning('Could not restore params')
            raise

    params = dict(
        parallel_rollout_manager_params=parallel_rollout_manager_params,
    )

    with open(os.path.join(jobdir, 'parallel_params.json'), 'w') as f:
        json.dump(params, f)

    return parallel_rollout_manager, parallel_train_manager, basic_conf, params, modules_fn, managed_memory, summary_writer

if __name__ == '__main__':
    from pprint import pprint

    params = dict(
        experiment='/local_data/sblaes/research/phd/repositories/public/rec_center/experiments/CWYC/experiments/pickandplace/three_task_her_upperBaseline.json',
        # json_string='{"jobdir":"/tmp/her_test",'
        #             '"params_path":"/home/sblaes/research/phd/repositories/public/rec_center/results/CWYC",'
        #             '"params_prefix":"latest"}'
        )

    basic_conf, modules_fn, \
    summary_writer = basic_configure(**params)

    pprint(basic_conf)

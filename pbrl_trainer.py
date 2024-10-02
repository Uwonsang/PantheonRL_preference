import argparse
import json
import gym
import yaml

from stable_baselines3 import PPO, PPO_REWARD
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from pantheonrl.common.wrappers import frame_wrap, recorder_wrap
from pantheonrl.common.agents import OnPolicyAgent

from pantheonrl.algos.modular.learn import ModularAlgorithm
from pantheonrl.algos.modular.policies import ModularPolicy

from pantheonrl.algos.bc import BCShell, reconstruct_policy

from overcookedgym.overcooked_utils import LAYOUT_LIST
from reward_model import RewardModel

ENV_LIST = ['OvercookedMultiEnv-v0']

EGO_LIST = ['PPO', 'PPO_REWARD', 'ModularAlgorithm', 'LOAD']


class EnvException(Exception):
    """ Raise when parameters do not align with environment """


def input_check(args):
    # Env checking
    if args.env == 'OvercookedMultiEnv-v0':
        if 'layout_name' not in args.env_config:
            raise EnvException(f"layout_name needed for {args.env}")
        elif args.env_config['layout_name'] not in LAYOUT_LIST:
            raise EnvException(
                f"{args.env_config['layout_name']} is not a valid layout")

    # Construct ego config
    if 'verbose' not in args.ego_config:
        args.ego_config['verbose'] = 1

    if (args.tensorboard_log is not None) != \
            (args.tensorboard_name is not None):
        raise EnvException("Must define log and names for tensorboard")


def generate_env(args):
    # TODO multi-processing & gpu-processing
    env = gym.make(args.env, **args.env_config, is_self_play=args.self_play)

    if args.framestack > 1:
        env = frame_wrap(env, args.framestack)

    if args.record is not None:
        env = recorder_wrap(env)

    return env


def generate_ego(env, args):
    kwargs = args.ego_config
    kwargs['env'] = env
    kwargs['device'] = args.device
    if args.seed is not None:
        kwargs['seed'] = args.seed

    kwargs['tensorboard_log'] = args.tensorboard_log

    ## TODO update kwargs ppo_reward
    '''
    tensorboard_log = args.tensorboard_log,
    learning_rate = args.lr,
    batch_size = args.batch_size,
    n_steps = args.n_steps,
    ent_coef = args.ent_coef,
    policy_kwargs = policy_kwargs,
    use_sde = use_sde,
    sde_sample_freq = args.sde_freq,
    gae_lambda = args.gae_lambda,
    clip_range = clip_range,
    n_epochs = args.n_epochs,
    num_interaction = args.re_num_interaction,
    num_feed = args.re_num_feed,
    feed_type = args.re_feed_type,
    re_update = args.re_update,
    metaworld_flag = metaworld_flag,
    max_feed = args.re_max_feed,
    unsuper_step = args.unsuper_step,
    unsuper_n_epochs = args.unsuper_n_epochs,
    size_segment = args.re_segment,
    max_ep_len = max_ep_len,
    verbose = 1
    '''

    if args.ego == 'LOAD':
        model = gen_load(kwargs, kwargs['type'], kwargs['location'])
        # wrap env in Monitor and VecEnv wrapper
        vec_env = DummyVecEnv([lambda: Monitor(env)])
        model.set_env(vec_env)
        if kwargs['type'] == 'ModularAlgorithm':
            model.policy.do_init_weights(init_partner=True)
            model.policy.num_partners = len(args.alt)
        return model
    elif args.ego == 'PPO':
        return PPO(policy='MlpPolicy', **kwargs)
    elif args.ego == 'PPO_REWARD':
        # instantiating the reward model
        reward_model = RewardModel(
            env.env.observation_space.shape[0],
            env.env.action_space.n,
            size_segment=args.re_segment,
            activation=args.re_act,
            lr=args.re_lr,
            mb_size=args.re_batch,
            teacher_beta=args.teacher_beta,
            teacher_gamma=args.teacher_gamma,
            teacher_eps_mistake=args.teacher_eps_mistake,
            teacher_eps_skip=args.teacher_eps_skip,
            teacher_eps_equal=args.teacher_eps_equal,
            large_batch=args.re_large_batch)
        return PPO_REWARD(reward_model, policy='MlpPolicy', **kwargs)
    elif args.ego == 'ModularAlgorithm':
        policy_kwargs = dict(num_partners=len(args.alt))
        return ModularAlgorithm(policy=ModularPolicy,
                                policy_kwargs=policy_kwargs,
                                **kwargs)
    else:
        raise EnvException("Not a valid policy")


def gen_load(config, policy_type, location):
    if policy_type == 'PPO':
        agent = PPO.load(location)
    elif policy_type == 'PPO_REWARD':
        agent = PPO_REWARD.load(location)
    elif policy_type == 'ModularAlgorithm':
        agent = ModularAlgorithm.load(location)
    elif policy_type == 'BC':
        agent = BCShell(reconstruct_policy(location))
    else:
        raise EnvException("Not a valid FIXED/LOAD policy")

    return agent


def generate_partners(env, ego, args):
    partners = []

    v = OnPolicyAgent(ego)
    env.add_partner_agent(v)
    partners.append(v)

    return partners


def preset(args, preset_id):
    '''
    helpful defaul configuration settings
    '''

    if preset_id == 1:
        env_name = args.env
        if 'layout_name' in args.env_config:
            env_name = "%s-%s" % (args.env, args.env_config['layout_name'])

        if args.tensorboard_log is None:
            args.tensorboard_log = 'logs'
        if args.tensorboard_name is None:
            args.tensorboard_name = '%s-%s-%d' % (
                env_name, args.ego, args.seed)
        if args.ego_save is None:
            args.ego_save = 'models/%s-%s-ego-%d' % (
                env_name, args.ego, args.seed)
        # if not args.record:
        #     args.record = 'trajs/%s-%s%s-%d' % (env_name, args.ego, args.alt[0], args.seed)
    else:
        raise Exception("Invalid preset id")
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
            Train ego and partner(s) in an environment.

            Environments:
            -------------
            All MultiAgentEnv environments are supported. Some have additional
            parameters that can be passed into --env-config. Specifically,
            OvercookedMultiEnv-v0 has a required layout_name parameter, so
            one must add:

                --env-config '{"layout_name":"[SELECTED_LAYOUT]"}'

            OvercookedMultiEnv-v0 also has parameters `ego_agent_idx` and
            `baselines`, but these have default initializations. LiarsDice-v0
            has an optional parameter, `probegostart`.

            The environment can be wrapped with a framestack, which transforms
            the observation to stack previous observations as a workaround
            for recurrent networks not being supported. It can also be wrapped
            with a recorder wrapper, which will write the transitions to the
            given file.

            Ego-Agent:
            ----------
            The ego-agent is considered the main agent in the environment.
            From the perspective of the ego agent, the environment functions
            like a regular gym environment.

            Supported ego-agent algorithms include PPO, ModularAlgorithm, ADAP,
            and ADAP_MULT. The default parameters of these algorithms can
            be overriden using --ego-config.
            
            NOTE:
            All configs are based on the json format, and will be interpreted
            as dictionaries for the kwargs of their initializers.

            Example usage (Overcooked with ADAP agents that share the latent
            space):

            python3 trainer.py OvercookedMultiEnv-v0 ADAP ADAP --env-config
            '{"layout_name":"random0"}' -l
            ''')

    parser.add_argument('env', default='OvercookedMultiEnv-v0',
                        choices=ENV_LIST,
                        help='The environment to train in')

    parser.add_argument('ego',
                        choices=EGO_LIST,
                        help='Algorithm for the ego agent')

    parser.add_argument('--total-timesteps', '-t',
                        type=int,
                        default=500000,
                        help='Number of time steps to run (ego perspective)')

    parser.add_argument('--device', '-d',
                        default='auto',
                        help='Device to run pytorch on')
    parser.add_argument('--seed', '-s',
                        default=None,
                        type=int,
                        help='Seed for randomness')

    parser.add_argument('--ego-config',
                        type=json.loads,
                        default={},
                        help='Config for the ego agent')

    parser.add_argument('--alt-config',
                        type=json.loads,
                        nargs='*',
                        help='Config for the ego agent')

    parser.add_argument('--env-config',
                        type=json.loads,
                        default={},
                        help='Config for the environment')

    parser.add_argument('--self_play', '-sp',
                        action='store_true',
                        help='self_play method with shared model')

    # Wrappers
    parser.add_argument('--framestack', '-f',
                        type=int,
                        default=1,
                        help='Number of observations to stack')

    parser.add_argument('--record', '-r',
                        help='Saves joint trajectory into file specified')

    parser.add_argument('--ego-save',
                        help='File to save the ego agent into')

    parser.add_argument('--tensorboard-log',
                        help='Log directory for tensorboard')

    parser.add_argument('--tensorboard-name',
                        help='Name for ego in tensorboard')

    parser.add_argument('--verbose-partner',
                        action='store_true',
                        help='True when partners should log to output')

    parser.add_argument('--preset', type=int, help='Use preset args')

    args = parser.parse_args()

    if args.preset:
        args = preset(args, args.preset)

    if args.ego == 'PPO_REWARD':
        args_dict = vars(args)
        with open("./config/pbrl_trainer.yaml") as f:
            ppo_reward_configs = yaml.load(f, Loader=yaml.FullLoader)
        combined_configs = {**args_dict, **ppo_reward_configs}
        args = argparse.Namespace(**combined_configs)

    input_check(args)

    print(f"Arguments: {args}")
    env = generate_env(args)
    print(f"Environment: {env}")
    ego = generate_ego(env, args)
    print(f'Ego: {ego}')
    partners = generate_partners(env, ego, args)

    learn_config = {'total_timesteps': args.total_timesteps}
    if args.tensorboard_log:
        learn_config['tb_log_name'] = args.tensorboard_name
    ego.learn(**learn_config)

    if args.record:
        transition = env.get_transitions()
        transition.write_transition(args.record)

    if args.ego_save:
        ego.save(args.ego_save)

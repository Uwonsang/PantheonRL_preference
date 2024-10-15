import gym

def ppo_make_overcooked_env(env_id, seed, layout_name, is_self_play):

    env = gym.make(env_id, **layout_name, is_self_play=is_self_play)
    env.seed(seed)

    return env
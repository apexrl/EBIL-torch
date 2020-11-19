envs_dict = {
    # "Standard" Mujoco Envs
    'halfcheetah': 'gym.envs.mujoco.half_cheetah:HalfCheetahEnv',
    'ant': 'gym.envs.mujoco.ant:AntEnv',
    'hopper': 'gym.envs.mujoco.hopper:HopperEnv',
    'walker': 'gym.envs.mujoco.walker2d:Walker2dEnv',
    'humanoid': 'gym.envs.mujoco.humanoid:HumanoidEnv',
    'swimmer': 'gym.envs.mujoco.swimmer:SwimmerEnv',
    'inverteddoublependulum': 'gym.envs.mujoco.inverted_double_pendulum:InvertedDoublePendulum2dEnv',
    'invertedpendulum': 'gym.envs.mujoco.inverted_pendulum:InvertedPendulum',

    # normal envs
    'lunarlandercont': 'gym.envs.box2d.lunar_lander:LunarLanderContinuous',

    # Envs we made for State-Marginal Matching
    'simple_point_mass': 'rlkit.envs.state_matching_point_mass_env:StateMatchingPointMassEnv',
    'pusher_trace_env': 'rlkit.envs.state_matching_pusher_env_no_obj:PusherTraceEnv',
    'pusher_smm_env': 'rlkit.envs.state_matching_pusher_env:PusherSMMEnv',
    'fetch_push_smm_env': 'rlkit.envs.state_matching_pickup_env:FetchPushSMMEnv',

    # Meta Environments

}

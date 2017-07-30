from gym.envs.registration import register

register(
    id='mikasa_gym-v0',
    entry_point='mikasa_gym.envs:MikasaEnv',
)
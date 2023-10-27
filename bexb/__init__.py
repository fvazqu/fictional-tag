from gymnasium.envs.registration import register
register(
         id='ball-v0',
         entry_point='bexa.bexb.envs.MyBallEnv:EnvBall',
         max_episode_steps=500,
         reward_threshold=3.75,
         )
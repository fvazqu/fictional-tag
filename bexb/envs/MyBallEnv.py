
from typing import Dict

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}
class EnvBall(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "fullpathtohelloworld.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 1,
        reward_touching_weight: float = 5,
        reward_fall_weight: float = -5,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_control_weight,
            reward_touching_weight,
            reward_fall_weight,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight
        self._reward_touching_weight = reward_touching_weight
        self._reward_fall_weight = reward_fall_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        vec = self.get_body_com("ball") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec) * self._reward_dist_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight


        vec2 = self.get_body_com("ball") - self.get_body_com("floor")

        # Checking if the ball has fallen
        if vec2[2] < -0.005:
           self.reset_model()
           reward_fall = self._reward_fall_weight
           #print('reset from fall')
           #print(reward_fall)
        else:
           reward_fall = 0.0
        

        # Reward for touching the target
        distance = np.linalg.norm(vec)
        if distance < 0.04:
            reward_touching = self._reward_touching_weight * self.steps_since_last_reset
        else:
            reward_touching = 0.0

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward = reward_dist + reward_ctrl + reward_touching + reward_fall
        info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_touching": reward_touching,
            "reward_fall": reward_fall,
        }

        # Check if the ball touches the target
        distance = np.linalg.norm(vec)
        if distance < 0.04:  # Adjust the threshold as needed
            self.steps_since_last_reset += 1
        else:
            self.steps_since_last_reset = 0

        if self.steps_since_last_reset >= 100:
            self.reset_model()  # Reset the model if the condition is met
            self.steps_since_last_reset = 0

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def reset_model(self):
        qpos = (
                self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
                + self.init_qpos
        )

        qpos[2:5] = [0, 0, 0]

        qvel = self.init_qvel + self.np_random.uniform(
            low=-3, high=3, size=self.model.nv
        )
        # Sets ball z vel to 0
        qvel[4] = 0

        # Sets ball x and y vel to 0
        qvel[2:4] = 0

        # Sets target x and y vel to 0
        qvel[0:2] = (0, 0)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # Assuming the x and y coordinates of the target are stored in qpos[2] and qpos[3]
        target_pos = self.data.qpos[0:2]
        #print(target_pos)

        # Assuming the x and y velocities of the ball are stored in qvel[0] and qvel[1]
        ball_velocity = self.data.qvel[2:5]
        #print(ball_velocity)

        # Assuming the x and y positions of the ball (agent) are stored in qpos[0] and qpos[1]
        ball_pos = self.data.qpos[2:5]
        #print(ball_pos)

        return np.concatenate([target_pos, ball_velocity, ball_pos])

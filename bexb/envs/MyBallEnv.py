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


    def __init__(
            self,
            xml_file: str = "C:/Users/fabian.DESKTOP-PATHTOG/PycharmProjects/MIRes/bexa/bexb/envs/modelos/helloworld.xml",
            frame_skip: int = 2,
            default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
            reward_vel_weight: float = 0.5,
            reward_control_weight: float = 0.05,
            reward_touching_weight: float = 1,
            reward_fall_weight: float = 0.5,
            **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_vel_weight,
            reward_control_weight,
            reward_touching_weight,
            reward_fall_weight,
            **kwargs,
        )

        self._reward_vel_weight = reward_vel_weight
        self._reward_control_weight = reward_control_weight
        self._reward_touching_weight = reward_touching_weight
        self._reward_fall_weight = reward_fall_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        self.steps_since_last_reset = 0

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

        # PART 1: UNIT VECTOR FOR POSITION
        # Calculate the magnitude of the vector
        mag_1 = np.linalg.norm(vec)

        # Calculate the unit vector
        unit_vector_1 = vec / mag_1

        #PART 2: UNIT VECTOR FOR VELOCITY
        obs = self._get_obs()
        vec_vel = obs[2:5]
        mag_2 = np.linalg.norm(vec_vel)
        unit_vector_2 = vec_vel / mag_2

        #PART 3: CHECK COS DIFFERENCE BETWEEN 1 & 2
        dot_product = np.dot(vec, vec_vel)
        cosine_angle = dot_product / (mag_1 * mag_2)

        reward_vel = -1*cosine_angle * self._reward_vel_weight
        #reward_dist = 0.5-(np.linalg.norm(vec) * self._reward_dist_weight)
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight

        vec2 = self.get_body_com("ball") - self.get_body_com("floor")

        # Checking if the ball has fallen
        if vec2[2] < -0.005:
            self.reset_model()
            reward_fall = -1*self._reward_fall_weight
            # print('reset from fall')
            # print(reward_fall)
        else:
            reward_fall = 0.0

        # Check if the ball touches the target
        distance = np.linalg.norm(vec)
        if distance < 0.05:  # Adjust the threshold as needed
            reward_touch = self._reward_touching_weight
            self.reset_model()  # Reset the model if the condition is met
        else:
            reward_touch = 0

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward = reward_vel + reward_ctrl + reward_touch + reward_fall

        info = {
            "reward_vel": reward_vel,
            "reward_ctrl": reward_ctrl,
            "reward_touch": reward_touch,
            "reward_fall": reward_fall,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def reset_model(self):
        qpos = (
                self.np_random.uniform(low=-0.35, high=0.35, size=self.model.nq)
                + self.init_qpos
        )

        #Spawn Box Randomly
        # random_y = random.uniform(-0.57, 0.37)
        # random_x = random.uniform(-.37, 0.57)
        # qpos[0] = 0.25
        # qpos[1] = 0

        # For testing, set ball to origin
        # qpos[2:4] = [0, 0]

        qpos[4] = 0
        #qpos[2:5] = [0, 0, 0]

        qvel = self.init_qvel + self.np_random.uniform(
            low=-3, high=3, size=self.model.nv
        )
        # Sets ball z vel to 0
        qvel[4] = 0

        # # Sets ball velocity
        # qvel[2] = 3
        # qvel[3] = 0

        # Sets target x and y vel to 0
        qvel[0:2] = (0, 0)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # Assuming the x and y coordinates of the target are stored in qpos[0] and qpos[1]
        target_pos = self.data.qpos[0:2]
        # print(target_pos)

        # Assuming the x and y velocities of the ball are stored in qvel[2] and qvel[3]
        ball_velocity = self.data.qvel[2:4]
        #print(ball_velocity)

        # Assuming the x and y positions of the ball (agent) are stored in qpos[2] and qpos[3]
        ball_pos = self.data.qpos[2:4]
        # print(ball_pos)

        return np.concatenate([target_pos, ball_velocity, ball_pos])

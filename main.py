import gymnasium as gym
import random
import bexb


def up():
   return [0, 1]

def down():
   return [0, -1]

def left():
   return [-1, 0]

def right():
   return [1, 0]

actions = [up(), down(), left(), right()]

steps_per_action = 100

env = gym.make("ball-v0", render_mode="human")
observation, info = env.reset(seed=42)

num_steps = 1000

for _ in range(num_steps):
   action = env.action_space.sample()  # this is where you would insert your policy
   action = random.choice(actions)

   #action = right()
   # action = left()
   # action = down()
   #action = up()
   #action[0] = 2.0 * (env.np_random.random() - 0.5)  # Random value between -1 and 1 for x
   #action[1] = 2.0 * (env.np_random.random() - 0.5)  # Random value between -1 and 1 for y

   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()

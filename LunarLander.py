# Using Gymnasium https://gymnasium.farama.org/environments/box2d/lunar_lander/

from Agent import Agent

import torch
from collections import deque, namedtuple
import numpy as np
import imageio
import gymnasium as gym
env = gym.make('LunarLander-v2')

###
### Gymnasium Paramters
###
# Vector of 8 inputs
state_shape = env.observation_space.shape
# The number of elemets in the input state
state_size = env.observation_space.shape[0]
# Action Space defined in gymnasum documentation
# 0: do nothing
# 1: fire left orientation engine
# 2: fire main engine
# 3: fire right orientation engine
number_actions = env.action_space.n


# Initialize the AI Agent
agent = Agent(state_size, number_actions)


#Train the Agent
number_episodes = 2000
maximum_number_timesteps_per_episode = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995 # Steps epsilon down to ending
epsilon = epsilon_starting_value
scores_on_100_episodes =  deque(maxlen = 100)
for episode in range(1, number_episodes + 1):
   state, _ = env.reset()
   score = 0 # Cumulative reward
   for time_step in range(maximum_number_timesteps_per_episode):
      action = agent.act(state, epsilon)
      next_state, reward, done, _, _ = env.step(action)
      agent.step(state, action, reward, next_state, done)
      state = next_state
      score += reward
      if done:
         break
   scores_on_100_episodes.append(score)
   epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

   print('\rEpisode {}\tAverageScore: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
   if episode % 100 == 0:
      print('\rEpisode {}\tAverageScore: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
   if np.mean(scores_on_100_episodes) >= 200.0:
      print('\nEnvironment solved in {:d} episodes!\tAverageScore: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
      torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
      break

# Create a mp4 visualization of the trained AI
def show_video_of_model(agent, env_name):
   env = gym.make(env_name, render_mode='rgb_array')
   state, _ = env.reset()
   done = False
   frames = []
   while not done:
      frame = env.render()
      frames.append(frame)
      action = agent.act(state)
      state, reward, done, _, _ = env.step(action.item())
   env.close()
   imageio.mimsave("video.mp4", frames, fps=30)

show_video_of_model(agent, 'LunarLander-v2')

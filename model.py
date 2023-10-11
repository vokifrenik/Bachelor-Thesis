# Import libraries

from PIL import Image           # Image management
import numpy as np              # Numerical computing
import matplotlib.pyplot as plt # Plotting

from nes_py.wrappers import JoypadSpace     # Joypad wrapper in NES Emulator

import gym_super_mario_bros                                 # Super Mario environment
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT    # Import the simplified controls


# Create environment - Note there are different environments, for more info: https://pypi.org/project/gym-super-mario-bros/
env = gym_super_mario_bros.make("SuperMarioBros-v2")

# Setup simplified controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# actions for simple movement
# SIMPLE_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
# ]



# Let us explore the space and action space in this environment

# Reset environment
env.reset()

# Sample one random action
action = env.action_space.sample()

# One agent-environment iteration 
next_state, reward, done, info = env.step(action)

# Print results
print('The shape of the states is {}'.format(next_state.shape))

print('The reward for this iteration is {}'.format(reward))

print('The done flag in this iteration is {}'.format(done))

print('Here is more information about this iteration {}'.format(info))


# Plot next state
plt.figure(figsize=(20, 16))
plt.imshow(next_state)
plt.show()


# Random Agent Loop

# Done flag - Termination of episode
done = True

# Number of iterations
num_iterations = 100

# Iteration loop
for steps in range(num_iterations):
    
    # Check if the episode is done
    if done:
        env.reset()
    
    # Sample one random action
    action = env.action_space.sample()
    
    # agent-environment iteration 
    next_state, reward, done, info = env.step(action)
    
    # Render may crash Jupyter
    env.render()
    
    # Check for new reward
    print('The reward in step {} is {}'.format(steps, reward))
    
# Close the environment        
env.close()

# Wrappers to vectorize and stack frames
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# RL algorithms in the library
from stable_baselines3 import DQN, A2C, DDPG, PPO, SAC, TD3

# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation



# Use the wrappers

# Grayscale
env = GrayScaleObservation(env, keep_dim=True)

# Vectorize environment - call the environment in sequence on the current Python process
env =  DummyVecEnv([lambda: env])

# Stack frames
env = VecFrameStack(env, n_stack = 4, channels_order='last')


# Explore the new environment

# Reset environment
env.reset()

# Sample one random action
action = env.action_space.sample()

# One agent-environment iteration 
next_state, reward, done, info = env.step([5])

# Print results
print('The shape of the states is {}'.format(next_state.shape))

print('The reward for this iteration is {}'.format(reward))

print('The done flag in this iteration is {}'.format(done))

print('Here is more information about this iteration {}'.format(info))


# Visualize the new state, i.e, past 4 frames
plt.figure(figsize=(20,16))
for idx in range(next_state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(next_state[0][:,:,idx])
plt.show()


# Create and train a model
model = PPO(policy = 'CnnPolicy', env = env, verbose = 1)
model.learn(total_timesteps = 1000)


# Start the game 
state = env.reset()
# Loop through the game
while True: 
    
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()

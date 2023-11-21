# environment.py

import numpy as np
import torch as T
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from actor_critic import Agent

# Setup environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Preprocess the environment
env = GrayScaleObservation(env, keep_dim=False)
env = FrameStack(env, 4)

input_size = env.observation_space.shape

# Create the Agent with the correct input_dims
agent = Agent(0.000005, 0.000001, input_dims = input_size, gamma=0.99, epsilon=0.2, n_actions=7, layer1_size=64, layer2_size=64, batch_size=64)

score_history = []
num_episodes = 100

for i in range(num_episodes):
    done = False
    score = 0
    state = env.reset()
    
    # Convert state to a single NumPy array
    state = np.array(state)
    
    # Convert the NumPy array to a PyTorch tensor
    state = T.tensor(state, dtype=T.float32).to(agent.actor.device)

    while not done:
        env.render()

        action = agent.choose_action(state)
        # Print the type of state and its dimensions
        #print(type(state))
        #print(state.shape)
        next_state, reward, done, info = env.step(action)

        # Convert next_state to a single NumPy array
        next_state = np.array(next_state)

        # Convert the NumPy array to a PyTorch tensor
        next_state = T.tensor(next_state, dtype=T.float32).to(agent.actor.device)

        agent.replay_buffer.store_transition(state, action, reward, next_state, done)


        score += reward
        print("score", score)

        if agent.replay_buffer.mem_cntr >= agent.replay_buffer.mem_size:
            state, action, reward, next_state, done = agent.replay_buffer.sample_buffer(agent.batch_size)
            print("state", state.shape)
            # Remove first dimension of state and next_state
            #state = state.squeeze(0)
            #next_state = next_state.squeeze(0)
            #print("state", state.shape)
            
            # Convert to PyTorch tensors
            #state = T.tensor(state, dtype=T.float32).to(agent.actor.device)
            #next_state = T.tensor(next_state, dtype=T.float32).to(agent.actor.device)
            #print("state", state.shape)
            #print("state type", type(state))
            agent.learn(state, reward, next_state, done)


        state = next_state

    env.close()

    score_history.append(score)
    print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

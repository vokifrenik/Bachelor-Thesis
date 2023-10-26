import numpy as np
import gym_super_mario_bros
import torch as T
from model import Agent
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT




# Setup environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Preprocess the environment
env = GrayScaleObservation(env, keep_dim=False)
env = FrameStack(env, 4)

state = env.reset()
state = state.__array__()
state = T.tensor(state, dtype=T.float32).to('cpu')


agent = Agent(0.000005, 0.000001, input_dims=state.shape, gamma=0.99, n_actions=7, layer1_size=64, layer2_size=64, n_outputs=2)

score_history = []
num_episodes = 100

for i in range(num_episodes):
    done = False
    score = 0
    state = env.reset()

    # convert state to tensor
    state = state.__array__()
    state = T.tensor(state, dtype=T.float32).to(agent.actor.device)

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        score += reward

        agent.learn(state, reward, next_state, done)
        state = next_state
    score_history.append(score)
    print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))


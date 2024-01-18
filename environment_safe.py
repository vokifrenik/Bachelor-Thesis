import os
import numpy as np
import torch as T
import gym
import gym_super_mario_bros
from icecream import ic
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from actor_critic_safe import Agent

# Setup environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Preprocess the environment
env = GrayScaleObservation(env, keep_dim=False)
env = FrameStack(env, 4)

input_size = env.observation_space.shape

# Create the Agent with the correct input_dims
agent = Agent(0.0000005, 0.0000001, input_dims=input_size, gamma=0.95, n_actions=7, layer1_size=64, layer2_size=64)

score_history = []
num_episodes = 15000

deaths = 0

for i in range(num_episodes):
    done = False
    score = 0
    current_episode_deaths = 0  # Counter for deaths in the current episode
    steps_since_last_score_increase = 0
    last_score = 0

    max_steps_without_increase = 100

    # obtain a state by resetting the environment
    obs = env.reset()

    # Convert state to a single NumPy array
    state = np.array(obs)

    # Convert the NumPy array to a PyTorch tensor
    state = T.tensor(state, dtype=T.float32).to(agent.actor.device)

    while not done:
        #env.render()

        action = agent.choose_action(state, 1)
        next_state, reward, done, info = env.step(action)
        #ic(info)
        #ic(state.shape)


        # get x-pos of mario from info
        x_pos = info['x_pos_screen']

        agent.get_distance(x_pos)

        # Check if Mario died
        if done:  
            current_episode_deaths += 1

        # Convert next_state to a single NumPy array
        next_state = np.array(next_state)

        # Convert the NumPy array to a PyTorch tensor
        next_state = T.tensor(next_state, dtype=T.float32).to(agent.actor.device)

        score += reward
        print("######################")
        print("SAFE")
        print("time", info['time'])
        print("action", action)
        print("score", score)
        print("reward", reward)
        print("done", done)
        print("deaths current episode", current_episode_deaths)
        print("episode", i)
        print("total deaths", deaths)
        print("world", info['world'])

        # Check if score has increased
        if score > last_score:
            last_score = score
            steps_since_last_score_increase = 0  
        else:
            steps_since_last_score_increase += 1

        agent.learn(state, reward, next_state, done)

        state = next_state

        # Reset if agent is stuck
        if steps_since_last_score_increase >= max_steps_without_increase:
            print("Mario's stuck")
            done = True

    deaths += current_episode_deaths  
    #why do you not see gumb_____________________________________________________________Miriow

    # Save the model every 20 episodes
    if i % 20 == 0:
        T.save({
            'epoch': i,
            'actor_weights': agent.actor_weights,
            'critic_weights1': agent.critic_weights1,
            'critic_weights2': agent.critic_weights2,
            'critic_weights3': agent.critic_weights3,
            'actor_optimizer': agent.actor_optimizer,
            'critic_optimizer1': agent.critic_optimizer1,
            'critic_optimizer2': agent.critic_optimizer2,
            'critic_optimizer3': agent.critic_optimizer3,
            'actor_loss': agent.actor_loss,
            'critic_loss1': agent.critic_loss1,
            'critic_loss2': agent.critic_loss2,
            'critic_loss3': agent.critic_loss3
            }, 'checkpoint.pth')


    score_history.append(score)
    print('episode ', i, 'score %.2f' % score, 'deaths in this episode', current_episode_deaths, 'total deaths', deaths,
          '100 game average %.2f' % np.mean(score_history[-100:]))
#death to mario
    
env.close()
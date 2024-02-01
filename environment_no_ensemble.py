import os
import numpy as np
import torch as T
import gym
import gym_super_mario_bros
#from icecream import ic
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from actor_critic_no_ensemble import Agent

# Setup environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Preprocess the environment
env = GrayScaleObservation(env, keep_dim=False)
env = FrameStack(env, 4)

input_size = env.observation_space.shape

# Create the Agent with the correct input_dims
agent = Agent(0.0000005, 0.0000001, input_dims=input_size, gamma=0.95, n_actions=7, layer1_size=64, layer2_size=64)

# Lists for storing scores and deaths
score_history = []
death_history = []

# Lists for storing total scores and deaths for every 100th episode
total_scores_100 = []
total_deaths_100 = []

num_episodes = 15000 # try 15K to 20K episodes

deaths = 0

for i in range(num_episodes):
    done = False
    score = 0
    current_episode_deaths = 0
    steps_since_last_score_increase = 0
    last_score = 0
    print(" for loop done", done)

    # Maximum allowed steps without score increase
    max_steps_without_increase = 100 

    obs = env.reset()
    print("do we get here?")
    state = np.array(obs)
    state = T.tensor(state, dtype=T.float32).to(agent.actor.device)

    while not done:
        action = agent.choose_action(state, 1)
        next_state, reward, done, info = env.step(action)

        if done:  
            current_episode_deaths += 1

        next_state = np.array(next_state)
        next_state = T.tensor(next_state, dtype=T.float32).to(agent.actor.device)

        score += reward
        #print("score", score, "reward", reward, "deaths in episode", current_episode_deaths)
        print("######################")
        print("NO ENSEMBLE")
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
        agent.record_weights()
        state = next_state

        # Reset if agent is stuck
        if steps_since_last_score_increase >= max_steps_without_increase:
            print("Mario's stuck")
            done = True

    print('episode ', i, 'score %.2f' % score, 'deaths in this episode', current_episode_deaths, 'total deaths', deaths)
          
    #why do you not see gumb_____________________________________________________________Miriow

    # Save the model every 20 episodes
    if i % 20 == 0:
        T.save({
            'epoch': i,
            'actor_weights': agent.actor_weights,
            'critic_weights1': agent.critic_weights1,
            'actor_optimizer': agent.actor_optimizer,
            'critic_optimizer1': agent.critic_optimizer1,
            'actor_loss': agent.actor_loss,
            'critic_loss1': agent.critic_loss1,
            }, 'checkpoint_no_ensemble.pth')
        
    deaths += current_episode_deaths
    score_history.append(score)
    death_history.append(current_episode_deaths)

    if i % 100 == 0 and i != 0:
        total_score_last_100 = sum(score_history[-100:])
        total_deaths_last_100 = sum(death_history[-100:])
        total_scores_100.append(total_score_last_100)
        total_deaths_100.append(total_deaths_last_100)


    score_history.append(score)
    print('episode ', i, 'score %.2f' % score, 'deaths in this episode', current_episode_deaths, 'total deaths', deaths,
          '100 game average %.2f' % np.mean(score_history[-100:]))
#death to mario
    
env.close()

with open('results_no_ensemble.txt', 'w') as file:
    file.write('Scores per Episode:\n')
    for score in score_history:
        file.write(str(score) + '\n')

    file.write('\nDeaths per Episode:\n')
    for death in death_history:
        file.write(str(death) + '\n')

    file.write('\nTotal Scores per 100 Episodes:\n')
    for total_score in total_scores_100:
        file.write(str(total_score) + '\n')

    file.write('\nTotal Deaths per 100 Episodes:\n')
    for total_death in total_deaths_100:
        file.write(str(total_death) + '\n')

    file.write('\nTotal Deaths:\n')
    file.write(str(deaths) + '\n')
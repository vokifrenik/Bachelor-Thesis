# model.py

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import random as rand
import numpy as np
import matplotlib.pyplot as plt

class GeneralNetwork(nn.Module):
   def __init__(self, lr, input_dims, fc1_dims, fc2_dims, output_dims):
        super(GeneralNetwork, self).__init__()

        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims

        # Convolutional layers with pooling
        self.conv1 = nn.Conv2d(input_dims, fc1_dims, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(fc1_dims, fc2_dims, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Modify the fully connected layers based on output_dims
        if output_dims == 2:
            self.fc = nn.Linear(3477, fc1_dims)
            self.relu_fc = nn.ReLU()
            self.mu_layer = nn.Linear(fc1_dims, 1)
            self.log_sigma_layer = nn.Linear(fc1_dims, 1)
        elif output_dims == 1:
            self.fc = nn.Linear(3477, fc2_dims)
            self.relu_fc = nn.ReLU()
            self.mu_layer = nn.Linear(fc2_dims, 1)
            self.log_sigma_layer = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = 'cpu'
        self.to(self.device)

   def forward_actor(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x_fc = self.relu_fc(self.fc(x))
        mu = self.mu_layer(x_fc).squeeze(-1)
        log_sigma = self.log_sigma_layer(x_fc).squeeze(-1)

        return mu, log_sigma

   def forward_critic(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x_fc = self.relu_fc(self.fc(x))
        value = self.mu_layer(x_fc).squeeze(-1)
        #print("value", value)

        return value
    
class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.95, epsilon=0.3, n_actions=7, layer1_size=64, layer2_size=64):
        self.input_dims = input_dims
        self.log_probs = None
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.actor = GeneralNetwork(alpha, input_dims, layer1_size, layer2_size, output_dims=2)
        self.critic = GeneralNetwork(beta, input_dims, layer1_size, layer2_size, output_dims=1)

    def choose_action(self, state):
        mu, log_sigma = self.actor.forward_actor(state)
        sigma = T.exp(log_sigma)

        # Sample from the normal distribution
        action_probs = T.distributions.Normal(mu, sigma)

        # Plot the distribution
        #plt.plot(action_probs.sample().numpy())
        #plt.show()

        # Sample only once and let the distribution broadcast across the batch dimension
        sampled_actions = action_probs.sample()

        # Calculate log probabilities for the sampled actions
        self.log_probs = action_probs.log_prob(sampled_actions).to(self.actor.device)

        # Epsilon-greedy strategy
        if rand.random() < self.epsilon:
            action = rand.randrange(self.n_actions)
        else:
            # Squeeze the mu tensor to get dimensions [64, 2] before finding the argmax
            action = T.argmax(mu.squeeze(-1), dim=-1).item()

        # Divide action by 10 to get the correct action and floor the value
        action = int(action / 10)
        print("action", action)
        return action

    def learn(self, state, reward, n_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        n_state_value = self.critic.forward_critic(n_state)
        state_value = self.critic.forward_critic(state)
        #print("n_state_value", n_state_value)
        #print("state_value", state_value)

        reward = T.tensor(reward, dtype=float).to(self.actor.device)
        
        delta = reward + self.gamma * n_state_value * (1 - int(done)) - state_value
        
        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2

        total_loss = actor_loss.mean() + critic_loss.mean()
        print("total_loss", total_loss)

        # Perform the backward pass on the scalar total_loss
        total_loss.backward()

        # Clip gradients to prevent exploding gradients
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        self.actor.optimizer.step()
        self.critic.optimizer.step()
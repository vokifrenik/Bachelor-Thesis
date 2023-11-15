# model.py

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import random as rand

class GeneralNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, output_dims):
        super(GeneralNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims

        # Convolutional layers with pooling
        self.conv1 = nn.Conv2d(input_dims[0], fc1_dims, kernel_size=3)
        print("input_dims[0]", input_dims[0])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(fc1_dims, fc2_dims, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Modify the fully connected layers based on output_dims
        if output_dims == 2:
            self.fc1 = nn.Linear(57 * 61, fc1_dims)
            self.relu_fc1 = nn.ReLU()
            self.fc2 = nn.Linear(fc1_dims, output_dims*2)
        elif output_dims == 1:
            self.fc1 = nn.Linear(57 * 61, fc2_dims)
            self.relu_fc1 = nn.ReLU()
            self.fc2 = nn.Linear(fc2_dims, output_dims*2)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)    
        x_fc1 = self.relu_fc1(self.fc1(x))
        x_fc2 = self.fc2(x_fc1)

        if x_fc2.size(-1) == 1:
            # Handle the case where x_fc2 has only one value
            mu, log_sigma = x_fc2, None
        else:
            # Unpack the result of chunking
            mu, log_sigma = T.chunk(x_fc2, 2, dim=-1)
        
        mu = mu.squeeze(-1)  # Remove the singleton dimension

        return mu, log_sigma
    

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
        mu, log_sigma = self.actor.forward(state)
        sigma = T.exp(log_sigma)

        print("mu", mu.size())
        print("sigma", sigma.size())

        # Sample from the normal distribution
        action_probs = T.distributions.Normal(mu, sigma)
        print(action_probs)

        # Sample only once and let the distribution broadcast across the batch dimension
        sampled_actions = action_probs.sample()

        # Use torch.clamp to ensure the sampled action values are within [0, 6]
        sampled_actions = T.clamp(sampled_actions, 0, 6)

        # Calculate log probabilities for the sampled actions
        self.log_probs = action_probs.log_prob(sampled_actions).to(self.actor.device)

        # Epsilon-greedy strategy
        if rand.random() < self.epsilon:
            action = rand.randrange(self.n_actions)
        else:
            # Squeeze the mu tensor to get dimensions [64, 2] before finding the argmax
            action = T.argmax(mu.squeeze(-1), dim=-1).item()

        print("action", action)
        return action

    def learn(self, state, reward, n_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        n_state_value, _ = self.critic.forward(n_state)
        state_value, _ = self.critic.forward(state)

        reward = T.tensor(reward, dtype=float).to(self.actor.device)
        
        delta = reward + self.gamma * n_state_value * (1 - int(done)) - state_value
        delta = delta.view(-1, 1, 1)  
        delta = delta.repeat(1, 7, 2)

        print("delta", delta.size())
        print("log_probs", self.log_probs.size())
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
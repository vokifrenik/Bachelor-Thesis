# model.py

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import random as rand
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init

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

        # Common fully connected layer
        ## Freeze this layer and then update the heads separaetely and vice versa
        self.fc = nn.Linear(3477, fc1_dims)
        self.relu_fc = nn.ReLU()

        # Actor head for mean and variance
        self.mu_layer = nn.Linear(fc1_dims, 1)
        self.log_sigma_layer = nn.Linear(fc1_dims, 1)  ## update 

        # Critic head for value function
        self.value_layer = nn.Linear(fc1_dims, 1)


        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = 'cpu'
        self.to(self.device)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_actor(self, x):
        if not isinstance(x, T.Tensor):
            x = T.tensor(x, dtype=T.float32).to(self.device)

        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x_fc = self.relu_fc(self.fc(x))
        ## Maybe add another relu after output
        mu = self.mu_layer(x_fc).squeeze(-1)
        log_sigma = self.log_sigma_layer(x_fc).squeeze(-1)

        return mu, log_sigma

    def forward_critic(self, x):
        # If x is not a tensor, convert it to a tensor
        if not isinstance(x, T.Tensor):
            x = T.tensor(x, dtype=T.float32).to(self.device)

        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x_fc = self.relu_fc(self.fc(x))
        value = self.value_layer(x_fc).squeeze(-1)

        return value
    
class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, epsilon=0.2, n_actions=7, layer1_size=64, layer2_size=64, batch_size=64):
        self.input_dims = input_dims
        self.log_probs = None
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.actor = GeneralNetwork(alpha, input_dims[0], layer1_size, layer2_size, output_dims=2)
        self.critic = GeneralNetwork(beta, input_dims[0], layer1_size, layer2_size, output_dims=1)
        self.replay_buffer = ReplayBuffer(100, input_dims, n_actions)


    def choose_action(self, state):
        mu, log_sigma = self.actor.forward_actor(state)
        sigma = T.exp(log_sigma)
        #sigma = T.abs(sigma)
        sigma = T.clamp(sigma, min=1e-6)  # Add a small epsilon to avoid zero
        
        #print("mu", mu)
        #print("sigma", sigma)

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
        

        # Check if there are any nan values in the loss tensors and replace them with small epsilon values
        for i in range(len(actor_loss)):
            if T.isnan(actor_loss[i]):
                actor_loss[i] = T.tensor(1e-6)

        for i in range(len(critic_loss)):
            if T.isnan(critic_loss[i]):
                critic_loss[i] = T.tensor(1e-6)

        #print("actor_loss", actor_loss)
        #print("critic_loss", critic_loss)


        total_loss = actor_loss.mean() + critic_loss.mean()  ## check this implementation
        print("total_loss", total_loss)

        # Perform the backward pass on the scalar total_loss
        total_loss.backward()

        # Clip gradients to prevent exploding gradients
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.actor.optimizer.step()
        self.critic.optimizer.step()



# The replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Initialize memory arrays to zeros
        self.state_memory = T.zeros((self.mem_size, *input_shape), dtype=T.float32)
        print("state_memory", self.state_memory.shape)
        self.new_state_memory = T.zeros((self.mem_size, *input_shape), dtype=T.float32)

        # Store actions as integers
        self.action_memory = T.zeros(self.mem_size, dtype=T.int64)

        # Store rewards as floats
        self.reward_memory = T.zeros(self.mem_size, dtype=T.float32)

        # Store done values as booleans
        self.terminal_memory = T.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, n_state, done):
        # Calculate the index to insert the experience
        index = self.mem_cntr % self.mem_size

        print("state in store_transition", state.shape)

        # Store the experience at the index
        self.state_memory[index] = state
        self.new_state_memory[index] = n_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # Increment the counter each time we store an experience
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # Calculate the maximum memory to sample from
        max_mem = min(self.mem_cntr, self.mem_size)

        # Generate a random batch of indices
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # Use the random indices to sample from memory
        states = self.state_memory[batch].squeeze(0)
        n_states = self.new_state_memory[batch].squeeze(0)  
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        print("state in sample_buffer", states.shape)

        # Return the batched experiences
        return states, actions, rewards, n_states, dones
    

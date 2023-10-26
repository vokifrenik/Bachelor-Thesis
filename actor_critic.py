import numpy as np
import torch as T
import torch.nn as nn
import torch.functional as F
import torch.optim as optim 

class GeneralNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, output_dims):
        super(GeneralNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims

        # Modify the convolutional layers to accept 3D input
        self.conv1 = nn.Conv2d(input_dims[0], fc1_dims, kernel_size=3)
        self.conv2 = nn.Conv2d(fc1_dims, fc2_dims, kernel_size=5)
        
        # Calculate the size of the linear layer based on the input dimensions
        linear_input_size = self._calculate_linear_input_size(input_dims)

        # Modify the fully connected layers based on output_dims
        if output_dims == 2:
            self.fc1 = nn.Linear(linear_input_size, fc1_dims)
            self.fc2 = nn.Linear(fc1_dims, output_dims)
        elif output_dims == 1:
            self.fc1 = nn.Linear(linear_input_size, fc2_dims)
            self.fc2 = nn.Linear(fc2_dims, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _calculate_linear_input_size(self, input_dims):
        dummy_input = T.zeros(1, *input_dims)
        x = nn.functional.relu(self.conv1(dummy_input))
        x = nn.functional.relu(self.conv2(x))
        return x.view(1, -1).size(1)


class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma = 0.95, n_actions=7, layer1_size=64, layer2_size=64, n_outputs=2):
        self.input_dims = input_dims
        self.log_probs = None
        self.gamma = gamma
        self.n_outputs = n_outputs
        self.actor = GeneralNetwork(alpha, input_dims, layer1_size, layer2_size, output_dims=2)
        self.critic = GeneralNetwork(beta, input_dims, layer1_size, layer2_size, output_dims=1)

    def choose_action(self, state):
        mu,sigma = self.actor.forward(state)
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))  
        self.log_probs = action_probs.log_probs(probs).to(self.actor.device)               

        return probs.item()
    

    def learn(self, state, reward, n_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        n_state_value = self.critic.forward(n_state)
        state_value = self.critic.forward(state)

        reward = T.tensor(reward, dtype=float).to(self.actor.device)
        delta = reward + self.gamma*n_state_value*(1-int(done)) - state_value

        actor_loss = -self.log_probs*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()



# model.py

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

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
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(fc1_dims, fc2_dims, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Modify the fully connected layers based on output_dims
        if output_dims == 2:
            self.fc1 = nn.Linear(57 * 61, fc1_dims)
            self.fc2 = nn.Linear(fc1_dims, output_dims)
        elif output_dims == 1:
            self.fc1 = nn.Linear(57 * 61, fc2_dims)
            self.fc2 = nn.Linear(fc2_dims, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        print("after cv1", x.shape)
        x = self.pool1(x)
        print("after pl1", x.shape)
        x = F.relu(self.conv2(x))
        print("after cv2", x.shape)
        x = self.pool2(x)
        print("after pl2", x.shape)
         # Flatten the output from conv layers
        x = x.view(x.size(0), -1)  
        print("after flattening", x.shape)
       
        x_fc1 = F.relu(self.fc1(x))
        print("after fc1", x.shape)
        x_fc2 = self.fc2(x_fc1)
        print("after fc2", x.shape)

         # Split the output into mu and sigma
        mu, log_sigma = T.chunk(x_fc2, 2, dim=-1)
        sigma = T.exp(log_sigma)

        return mu, sigma
    

class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.95, n_actions=7, layer1_size=64, layer2_size=64):
        self.input_dims = input_dims
        self.log_probs = None
        self.gamma = gamma
        self.actor = GeneralNetwork(alpha, input_dims, layer1_size, layer2_size, output_dims=2)
        self.critic = GeneralNetwork(beta, input_dims, layer1_size, layer2_size, output_dims=1)

    def choose_action(self, state):
        mu, sigma = self.actor.forward(state)
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)

        # Sample actions and get their probabilities
        sample_shape = T.Size([self.actor.output_dims])
        sampled_actions = action_probs.sample(sample_shape).squeeze()
        self.log_probs = action_probs.log_prob(sampled_actions).to(self.actor.device)

        # Choose the action with the highest probability
        action = T.argmax(self.log_probs).item()

        # If you want to sample an action, uncomment the next line:
        # action = T.multinomial(self.log_probs.exp(), 1).item()
        return action


    def learn(self, state, reward, n_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        n_state_value = self.critic.forward(n_state)
        state_value = self.critic.forward(state)

        reward = T.tensor(reward, dtype=float).to(self.actor.device)
        delta = reward + self.gamma * n_state_value * (1 - int(done)) - state_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()
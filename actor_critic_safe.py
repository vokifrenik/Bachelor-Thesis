import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import random as rand
import numpy as np
from torch.nn import init
import cv2
#from icecream import ic
import matplotlib.pyplot as plt
from PIL import Image
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def find_object(self, state):
    # Load the images
    # Convert the first frame of the tensor to numpy array
    #state = state[0].cpu().detach().numpy()
    state = state[0].cuda().detach().cpu().numpy()

    # Convert NumPy array to image
    #large_image_gray = Image.fromarray(state)
    #ic(large_image_gray)
    #small_image = cv2.imread('C:\Bachelor Thesis\Bachelor-Thesis\images\GuillaumeGoomb.png')
    small_image = cv2.imread('GuillaumeGoomb.PNG')

    # Convert to grayscale
    small_image_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    
    # convert small_image_gray to uint8
    small_image_gray = small_image_gray.astype(np.uint8)
    state = state.astype(np.uint8)

    # Template matching
    result = cv2.matchTemplate(state, small_image_gray, cv2.TM_CCOEFF_NORMED)

    # Set a threshold
    threshold = 0.47

    # Find where the match is
    locations = np.where(result >= threshold)

    # Extract the coordinates of the match
    locations = list(zip(*locations[::-1]))

    # Draw bounding box
    w, h = small_image_gray.shape[::-1]  # Width and height of the template
    for loc in locations:
        top_left = (loc[1], loc[0])  # (x, y) coordinates
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(state, top_left, bottom_right, (0, 255, 0), 2)

    

    # If it exists return coordinates otherwise return None
    if locations:
        self.goomba = locations[0]
        # Show the image
        #plt.imshow(state)
        return locations[0]
    else:
        self.goomba = None
        return None

    
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
        ## Freeze this layer and then update the heads separately and vice versa
        self.fc = nn.Linear(3477 + 1, fc1_dims)
        # self.fc = nn.Linear(fc2_dims * 1 * 1, fc1_dims)
        self.relu_fc = nn.ReLU()

        # Actor head for mean and variance
        self.mu_layer = nn.Linear(fc1_dims, 1)
        self.log_sigma_layer = nn.Linear(fc1_dims, 1)  ## update here with softmax?

        # Critic head for value function
        self.value_layer = nn.Linear(fc1_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = 'cuda'
        self.to(self.device)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_actor(self, x, distance):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        # Handle distance - reshape and concatenate
        # Check if distance is None and replace it with a default value (e.g., a large number)
        if distance is None:
            distance = 1000  # or any other value that makes sense in your context
        distance = T.tensor([distance], dtype=T.float32).to(self.device)
        
           # Convert distance to a tensor and adjust its dimensions
        distance = distance.view(-1, 1)  # Reshape to [batch_size, 1]
    
        # Make sure distance has the same batch size as x
        if distance.size(0) != x.size(0):
            distance = distance.expand(x.size(0), -1)
        #ic(distance.shape)
        #ic(x.shape)
        # Concatenate the flattened image tensor and the distance tensor
        x_fc = T.cat((x, distance), dim=1)

        x_fc = F.relu(self.fc(x_fc))
        mu = self.mu_layer(x_fc).squeeze(-1)
        log_sigma = self.log_sigma_layer(x_fc).squeeze(-1)
        log_sigma = T.clamp(log_sigma, min=-20, max=2)
        log_sigma = log_sigma + 1e-6
        sigma = T.exp(log_sigma)

        return mu, sigma

    def forward_critic(self, x, distance):
        if not isinstance(x, T.Tensor):
            x = T.tensor(x, dtype=T.float32).to(self.device)

        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        # Handle distance - reshape and concatenate
        # Check if distance is None and replace it with a default value
        if distance is None:
            distance = 1000  # or any other value that makes sense in your context
        distance = T.tensor([distance], dtype=T.float32).to(self.device)

        # Reshape to match batch size
        distance = distance.view(-1, 1)  
        # Make sure distance has the same batch size as x
        if distance.size(0) != x.size(0):
            distance = distance.expand(x.size(0), -1)
        #ic(distance.shape)
        #ic(x.shape)
        x_fc = T.cat((x, distance), dim=1)  # Concatenate distance with the conv output

        x_fc = self.relu_fc(self.fc(x_fc))
        value = self.value_layer(x_fc).squeeze(-1)

        return value


class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.90, n_actions=7, layer1_size=64, layer2_size=64):
        self.input_dims = input_dims
        self.log_probs = None

        self.actor_loss = None
        self.critic_loss1 = None
        self.critic_loss2 = None
        self.critic_loss3 = None

        self.actor_weights = None
        self.critic_weights1 = None
        self.critic_weights2 = None
        self.critic_weights3 = None

        self.actor_optimizer = None
        self.critic_optimizer1 = None
        self.critic_optimizer2 = None
        self.critic_optimizer3 = None

        self.goomba = None
        self.distance = None
        self.gamma = gamma
        self.n_actions = n_actions
        self.actor = GeneralNetwork(alpha, input_dims[0], layer1_size, layer2_size, output_dims=2)
        self.critic1 = GeneralNetwork(beta, input_dims[0], layer1_size, layer2_size, output_dims=1)
        self.critic2 = GeneralNetwork(beta, input_dims[0], layer1_size, layer2_size, output_dims=1)
        self.critic3 = GeneralNetwork(beta, input_dims[0], layer1_size, layer2_size, output_dims=1)

    def choose_action(self, state, temperature):
        mu, sigma = self.actor.forward_actor(state, self.distance)

        # Calculate the variance
        sigma_sq = sigma ** 2

        # Sample from the normal distribution
        #action_probs = T.distributions.Normal(mu, sigma_sq)

        # Remove negative values from the distribution
        #action_probs = T.clamp(action_probs, min=0)

        # Create a normal distribution
        normal_distribution = Normal(loc=mu, scale=sigma_sq)

        # Sample values from the normal distribution
        samples = normal_distribution.sample((7,))

        # Calculate the log probabilities of the actions
        self.log_probs = normal_distribution.log_prob(samples)

        # Boltzmann exploration
        action_probs_softmax = F.softmax(mu / temperature, dim=-1) 
        action_distribution = T.distributions.Categorical(action_probs_softmax)  
        action = action_distribution.sample().item() 

        # Divide action by 10 to get the correct action and floor the value
        action = int(action / 10)

        find_object(self, state)

        #ic(goomba)
        #ic(action)

        # Reduce temperature to get more greedy 
        temperature = temperature * 0.9999

        return action

    def learn(self, state, reward, n_state, done):
        self.actor.optimizer.zero_grad()
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        self.critic3.optimizer.zero_grad()

        n_state_value1 = self.critic1.forward_critic(n_state, self.distance)
        n_state_value2 = self.critic2.forward_critic(n_state, self.distance)
        n_state_value3 = self.critic3.forward_critic(n_state, self.distance)

        state_value1 = self.critic1.forward_critic(state, self.distance)
        state_value2 = self.critic2.forward_critic(state, self.distance)
        state_value3 = self.critic3.forward_critic(state, self.distance)

        # Calculate separate delta and loss for each critic
        delta1 = (reward + self.gamma * n_state_value1 * (1 - int(done)) - state_value1).mean()
        delta2 = (reward + self.gamma * n_state_value2 * (1 - int(done)) - state_value2).mean()
        delta3 = (reward + self.gamma * n_state_value3 * (1 - int(done)) - state_value3).mean()

        self.log_probs = self.log_probs.masked_fill(T.isnan(self.log_probs), 1e-6)

        # Use clone to avoid in-place modifications
        actor_loss = -self.log_probs * delta1
        critic_loss1 = delta1 ** 2

        actor_loss += -self.log_probs * delta2
        critic_loss2 = delta2 ** 2

        actor_loss += -self.log_probs * delta3
        critic_loss3 = delta3 ** 2

        # Calculate uncertainties
        actor_uncertainty = self.log_probs.var()
        actor_uncertainty = T.clamp(actor_uncertainty, max=1e6)  # Add a maximum value to prevent instability

        # Define weights for the uncertainty terms
        actor_uncertainty_weight = 0.01
        critic_uncertainty_weight = 0.01

        # Add uncertainty terms to the losses
        actor_loss += actor_uncertainty_weight * actor_uncertainty
        critic_loss1 += critic_uncertainty_weight * state_value1.var()
        critic_loss2 += critic_uncertainty_weight * state_value2.var()
        critic_loss3 += critic_uncertainty_weight * state_value3.var()

        # Handle NaN values
        actor_loss = actor_loss.masked_fill(T.isnan(actor_loss), 1e-6)
        critic_loss1 = critic_loss1.masked_fill(T.isnan(critic_loss1), 1e-6)
        critic_loss2 = critic_loss2.masked_fill(T.isnan(critic_loss2), 1e-6)
        critic_loss3 = critic_loss3.masked_fill(T.isnan(critic_loss3), 1e-6)

        # Clip gradients to prevent exploding gradients
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        T.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=0.5)
        T.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=0.5)
        T.nn.utils.clip_grad_norm_(self.critic3.parameters(), max_norm=0.5)

        # Turn to scalar
        actor_loss = actor_loss.mean()
        critic_loss1 = critic_loss1.mean()
        critic_loss2 = critic_loss2.mean()
        critic_loss3 = critic_loss3.mean()

        self.actor_loss = actor_loss
        self.critic_loss1 = critic_loss1
        self.critic_loss2 = critic_loss2
        self.critic_loss3 = critic_loss3

        # Perform the backward pass on the scalar actor_loss
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Zero out the gradients to avoid accumulation
        self.actor.optimizer.zero_grad()

        # Perform the backward pass on the scalar critic_loss
        critic_loss1.backward(retain_graph=True)
        self.critic1.optimizer.step()

        critic_loss2.backward(retain_graph=True)
        self.critic2.optimizer.step()

        critic_loss3.backward(retain_graph=True)
        self.critic3.optimizer.step()

    def get_distance(self, x_pos):
        # If the goomba is not found, return None
        if self.goomba is None:
            return None

        # Calculate the distance between Mario and the goomba
        # This calculates the wrong distance for some reason, maybe using wrong coordinates?
        # Also need to add maybe a self.distance to use it to feed into the dense layers but thats for later
        distance = abs(x_pos - self.goomba[0])
        #ic(x_pos)
        #ic(self.goomba[0])
        #ic(distance)

        self.distance = distance
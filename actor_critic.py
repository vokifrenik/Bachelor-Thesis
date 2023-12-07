import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import random as rand
import numpy as np
from torch.nn import init
import pyautogui
import cv2
from icecream import ic
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

        # Common fully connected layer
        ## Freeze this layer and then update the heads separately and vice versa
        self.fc = nn.Linear(3477, fc1_dims)
        # self.fc = nn.Linear(fc2_dims * 1 * 1, fc1_dims)
        self.relu_fc = nn.ReLU()

        # Actor head for mean and variance
        self.mu_layer = nn.Linear(fc1_dims, 1)
        self.log_sigma_layer = nn.Linear(fc1_dims, 1)  ## update here with softmax?

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
        #ic(x)
        x = F.relu(self.conv1(x))
        #ic(x)
        x = self.pool1(x)
        #ic(x)
        x = F.relu(self.conv2(x))
        #ic(x)
        x = self.pool2(x)
        #ic(x)
        x = x.view(x.size(0), -1)
        #ic(x)
        x_fc = F.relu(self.fc(x))
        #ic(x_fc)
        mu = self.mu_layer(x_fc).squeeze(-1)
        #ic(mu)
        log_sigma = self.log_sigma_layer(x_fc).squeeze(-1)

        # Make sure the log_sigma is not too large or too small
        log_sigma = T.clamp(log_sigma, min=-20, max=2)
        # Add a small value to prevent log(0)
        log_sigma = log_sigma + 1e-6

        sigma = T.exp(log_sigma)
        #ic(sigma)

        return mu, sigma

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
        #ic(value)
        return value


class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, epsilon=0.2, n_actions=7, layer1_size=64, layer2_size=64):
        self.input_dims = input_dims
        self.log_probs = None
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.actor = GeneralNetwork(alpha, input_dims[0], layer1_size, layer2_size, output_dims=2)
        self.critic = GeneralNetwork(beta, input_dims[0], layer1_size, layer2_size, output_dims=1)
    
        # Initialize the GameClassifier with paths to the Goomba and cliff pattern images
        self.classifier = GameClassifier(goomba_image_path='C:\Bachelor Thesis\Bachelor-Thesis\images\goomba.png',
                                         cliff_pattern_image_path='C:\Bachelor Thesis\Bachelor-Thesis\images\cliff.png')

    def choose_action(self, state, temperature=1):
        mu, sigma = self.actor.forward_actor(state)

        # Calculate the variance
        sigma_sq = sigma ** 2

        # Sample from the normal distribution
        action_probs = T.distributions.Normal(mu, sigma_sq)

        # Sample only once and let the distribution broadcast across the batch dimension
        sampled_actions = action_probs.sample()

        # Calculate log probabilities for the sampled actions
        self.log_probs = action_probs.log_prob(sampled_actions).to(self.actor.device)

        # Boltzmann exploration
        action_probs_softmax = F.softmax(mu / temperature, dim=-1) # Divide by temperature to control exploration
        action_distribution = T.distributions.Categorical(action_probs_softmax)  # Makes categorical disribution of actions
        action = action_distribution.sample().item() # Samples an action from the distribution

        # Divide action by 10 to get the correct action and floor the value
        action = int(action / 10)

        # Make an image of the current state's first frame
        current_state_image = state[0].cpu().detach().numpy()

        # display the image
        ic(current_state_image.shape)


        # Check if a Goomba is present in the current state
        is_goomba_present = self.classifier.is_goomba_present(current_state_image)

        # Check if a cliff is ahead in the current state
        is_cliff_ahead = self.classifier.is_cliff_ahead(current_state_image)

        # Print the results
        if is_goomba_present:
            print("Goomba is present!")
        else:
            print("No Goomba detected.")

        if is_cliff_ahead:
            print("Cliff is ahead!")
        else:
            print("No cliff detected.")

        return action

    def learn(self, state, reward, n_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        n_state_value = self.critic.forward_critic(n_state)
        state_value = self.critic.forward_critic(state)

        delta = reward + self.gamma * n_state_value * (1 - int(done)) - state_value

        self.log_probs = self.log_probs.masked_fill(T.isnan(self.log_probs), 1e-6)

        # Use clone to avoid in-place modifications
        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2

        # Calculate uncertainties
        actor_uncertainty = self.log_probs.var()
        actor_uncertainty = T.clamp(actor_uncertainty, max=1e6)  # Add a maximum value to prevent instability
        critic_uncertainty = state_value.var()

        # Define weights for the uncertainty terms
        actor_uncertainty_weight = 0.01
        critic_uncertainty_weight = 0.01

        # Add uncertainty terms to the losses
        actor_loss += actor_uncertainty_weight * actor_uncertainty
        critic_loss += critic_uncertainty_weight * critic_uncertainty

        # Handle NaN values
        actor_loss = actor_loss.masked_fill(T.isnan(actor_loss), 1e-6)
        critic_loss = critic_loss.masked_fill(T.isnan(critic_loss), 1e-6)

        # Clip gradients to prevent exploding gradients
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        # Calculate total loss
        total_loss = actor_loss + critic_loss

        # Handle NaN values in the total loss
        total_loss = total_loss.masked_fill(T.isnan(total_loss), 1e-6)
        total_loss = total_loss.mean()

        # Perform the backward pass on the scalar total_loss
        total_loss.backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()


class GameClassifier:
    def __init__(self, goomba_image_path, cliff_pattern_image_path, screenshot_dimensions=(1920, 1080)):
        self.screenshot_dimensions = screenshot_dimensions
        self.goomba_image = cv2.imread(goomba_image_path)
        self.cliff_pattern_image = cv2.imread(cliff_pattern_image_path)

    def get_game_screenshot(self):
        # Take screenshot of state
        screenshot = pyautogui.screenshot()

        # Convert the PIL Image to a NumPy array
        screenshot_np = np.array(screenshot)

        # Resize the screenshot to the desired dimensions
        screenshot_np = self._resize_screenshot(screenshot_np)

        return screenshot_np

    def _resize_screenshot(self, screenshot_np):
        # Resize the screenshot to the desired dimensions
        screenshot_np = cv2.resize(screenshot_np, self.screenshot_dimensions)

        return screenshot_np

    def is_goomba_present(self, current_state):
        if self.goomba_image is None:
            print("Error: Goomba image not provided.")
            return False
        
        # Turn the goomba image into grayscale
        self.goomba_image = cv2.cvtColor(self.goomba_image, cv2.COLOR_BGR2GRAY)
        
        ic(self.goomba_image.shape)
       

        # Resize the current state to the same dimensions as the goomba image
        current_state_resized = cv2.resize(current_state, self.goomba_image.shape[:2][::-1])
        ic(current_state_resized.shape)

        # Compute Mean Squared Error (MSE) between the two images
        mse = np.sum((current_state_resized - self.goomba_image) ** 2) / float(current_state.size)

        # Revert the goomba image to BGR    
        self.goomba_image = cv2.cvtColor(self.goomba_image, cv2.COLOR_GRAY2BGR)

        # You can define a threshold for MSE to determine if the images are similar
        threshold = 500
        return mse < threshold

    def is_cliff_ahead(self, current_state):
        if self.cliff_pattern_image is None:
            print("Error: Cliff pattern image not provided.")
            return False
        
        # Turn the cliff pattern image into grayscale
        self.cliff_pattern_image = cv2.cvtColor(self.cliff_pattern_image, cv2.COLOR_BGR2GRAY)

        # Resize the current state to the same dimensions as the cliff pattern image
        current_state_resized = cv2.resize(current_state, self.cliff_pattern_image.shape[:2][::-1])

        # Compute Mean Squared Error (MSE) between the two images
        mse = np.sum((current_state_resized - self.cliff_pattern_image) ** 2) / float(current_state.size)

        # Revert the cliff pattern image to BGR
        self.cliff_pattern_image = cv2.cvtColor(self.cliff_pattern_image, cv2.COLOR_GRAY2BGR)

        # You can define a threshold for MSE to determine if the images are similar
        threshold = 500
        return mse < threshold

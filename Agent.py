import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F

from NeuralNetwork import NeuralNetwork
from ReplayMemory import ReplayMemory

###
### hyperparameters
###
learning_rate = 5e-4
# number of observations in one step of the training to update the model parameters
minibatch_size = 100
# Present value of future rewards
# Closer to 0 is short sited and only considers current rewards, close to 1 considers future rewards
discount_factor = 0.99 # aka gamma
# Memory of the IA, number of experences to keep
replay_buffer_size = int(1e5) # 100,000
# Used in the subupdate of the target network
interpolation_parameter = 1e-3 # 0.003

class Agent(): # Uses Deep Q Network
   def __init__(self, state_size, action_size):
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.state_size = state_size
      self.action_size = action_size

      # Q-Learning
      # Selects the actions
      self.local_qnetwork = NeuralNetwork(state_size, action_size).to(self.device) 
      # Calculates the target q values that will be used in the training of the local network
      self.target_qnetwork = NeuralNetwork(state_size, action_size).to(self.device) 
      # Takes the weight of the network and the learning rate
      self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate) 
      self.memory = ReplayMemory(replay_buffer_size)
      self.time_step = 0

   # Store experiences and decides when to learn from them
   def step(self, state, action, reward, next_state, done): # Decomposed parts of experience
      self.memory.push((state, action, reward, next_state, done))
      # Incrememnt counter and reset it every 4 steps so that we can learn every 4 steps
      self.time_step = (self.time_step + 1) % 4
      if self.time_step == 0:
         if len(self.memory.memory) > minibatch_size: # Check to see if our minibatch is over 100
            experiences = self.memory.sample(minibatch_size)
            self.learn(experiences, discount_factor)

   # Select an action based on a given state and epsilon value for an epsilon greedy action selection policy
   def act(self, state, epsilon=0.):
      # Convert to torch tensor and add an extra dimension to our state vector that corresponds to the batch
      state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
      # Set local network to evaluation mode
      self.local_qnetwork.eval()
      # Pass the state through the network
      with torch.no_grad(): # gradients are disabled
         action_values = self.local_qnetwork(state)
      # Go back into training mode
      self.local_qnetwork.train()

      # Greedy Epsilon 
      if random.random() > epsilon:
         return np.argmax(action_values.cpu().data.numpy()) # Selects action with the highest Q value
      else:
         return random.choice(np.arange(self.action_size)) # Select random from possible. Lets the Agent explore other options

   # Update agents Q values based on sampled experiences
   def learn(self, experiences, discount_facotr):
      # Extract elements from experience
      states, next_states, actions, rewards, dones = experiences
      # Get the maximum prodicted Q values for the next states
      next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1) # Maximum along dimension 1
      # Q targets for our current state
      q_targets = rewards + (discount_facotr * next_q_targets * (1-dones))
      # Expected Q values from our local network
      q_expected = self.local_qnetwork(states).gather(1, actions)
      # Loss between actual and expected
      loss = F.mse_loss(q_expected, q_targets) # Meas squared error loss
      
      # Back propagate this error loss
      self.optimizer.zero_grad() # Reset optimizer
      loss.backward()
      self.optimizer.step() # Update parameters of model

      # Update target network parameters with that of local
      self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

   # Update target network parameters with that of local
   def soft_update(self, local_model, target_model, interpolation_parameter):
      for target_paramaters, local_parameters in zip(target_model.parameters(), local_model.parameters()):
         target_paramaters.data.copy_(interpolation_parameter * local_parameters.data + (1.0 - interpolation_parameter) * target_paramaters.data)

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
   def __init__(self, state_size, action_size, seed=42):
      super(NeuralNetwork, self).__init__()
      self.seed = torch.manual_seed(seed)

      # Number of Neurons used per layer of the network
      numberOfNeurons = 64

      # First full connection between the input layer and the first fully connected layer
      # First paramater is number of neurons in the input layer, second is number of neurons in fully connected layer
      self.fc1 = nn.Linear(state_size, numberOfNeurons)
      self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
      self.fc3 = nn.Linear(numberOfNeurons, action_size) # action_size is output layer

   # Propagate the signal from the input layer to the output through the fully connected layers
   def forward(self, state):
      # Go through first layer
      output = self.fc1(state)
      output = F.relu(output) # Rectifier activation function

      # Go through second layer
      output = self.fc2(output)
      output = F.relu(output) # Rectifier activation function

      # Go through final layer
      return self.fc3(output)

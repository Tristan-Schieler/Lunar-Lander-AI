import torch
import random
import numpy as np

class ReplayMemory(object):
   def __init__(self, capacity):
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.capacity = capacity # Maximum size of the memory buffer
      self.memory = [] # Stores experiences
   
   # Add experience to memory buffer
   def push(self, experience):
      self.memory.append(experience)

      # Check to see if we are past capacity and remove oldest memory if we are
      if len(self.memory) > self.capacity:
         del self.memory[0]

   # randomly select a batch of memorys
   def sample(self, batch_size):
      experiences = random.sample(self.memory, batch_size)
      
      # Stack all the sates from the experiences
      states = np.vstack([experience[0] for experience in experiences if experience is not None])
      # Convert states into pytorch tensors and ensure they are floats
      states = torch.from_numpy(states).float().to(self.device)
      # Do the same with other parts of experience
      actions = torch.from_numpy(np.vstack([experience[1] for experience in experiences if experience is not None])).long().to(self.device)
      rewards = torch.from_numpy(np.vstack([experience[2] for experience in experiences if experience is not None])).float().to(self.device)
      next_states = torch.from_numpy(np.vstack([experience[3] for experience in experiences if experience is not None])).float().to(self.device)
      # npuint8 converting boolean to float
      dones = torch.from_numpy(np.vstack([experience[4] for experience in experiences if experience is not None]).astype(np.uint8)).float().to(self.device)

      return states, next_states, actions, rewards, dones

import random 
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        """
        Initialize the replay buffer.
        
        Args:
            max_size (int): Maximum number of experiences to store.
            input_shape (tuple): Shape of the input observation.
        """
        self.buffer = deque(maxlen=max_size)
        self.input_shape = input_shape
    
    def store(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size: Number of transitions to sample.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32), 
            np.array(actions, dtype=np.int32), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states, dtype=np.float32), 
            np.array(dones, dtype=np.float32)
        )
    
    def size(self):
        """
        Return the current number of stored transitions.
        """
        return len(self.buffer)

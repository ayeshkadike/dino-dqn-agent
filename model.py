# This is a DQN agent implementation using tensorflow and keras that will be used to train the agent to play the dino game.

import tensorflow as tf
from tensorflow.keras import layers, models

def build_dqn_model(input_shape=(84,84,4), num_actions = 3):
    
    """
    Builds a Deep Q-Network model.

    Args:
        input_shape: Shape of the observation (e.g., (84, 84, 4))
        num_actions: Number of possible actions

    Returns:
        A compiled tf.keras.Model
    """
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(
        filters=32, kernel_size=8, strides=4, activation='relu',
        input_shape=input_shape
    ))

    
    model.add(layers.Conv2D(
        filters=64, kernel_size=4, strides=2, activation='relu'
    ))
    
    model.add(layers.Conv2D(
        filters=64, kernel_size=3, strides=1, activation='relu'
    ))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    
    return model


model = build_dqn_model()
model.summary()




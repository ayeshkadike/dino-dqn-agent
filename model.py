import tensorflow as tf
from tensorflow.keras import layers, models

def build_dqn_model(input_shape=(84, 84, 4), num_actions=3):
    inputs = layers.Input(shape=input_shape)

    # Feature extraction
    x = layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', kernel_initializer='glorot_uniform')(inputs)
    x = layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = layers.Flatten()(x)

    # Dueling DQN
    # Value stream
    value = layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform')(x)
    value = layers.Dense(1, kernel_initializer='glorot_uniform')(value)

    # Advantage stream
    advantage = layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform')(x)
    advantage = layers.Dense(num_actions, kernel_initializer='glorot_uniform')(advantage)

    # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
    mean_advantage = layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
    q_values = layers.Add()([value, layers.Subtract()([advantage, mean_advantage])])

    model = models.Model(inputs=inputs, outputs=q_values)
    return model

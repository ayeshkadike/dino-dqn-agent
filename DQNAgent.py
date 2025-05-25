import tensorflow as tf
import numpy as np

class DQNAgent:
    def __init__(self, model, target_model, num_actions, gamma=0.99, lr=1e-4):
        self.model = model
        self.target_model = target_model
        self.num_actions = num_actions
        self.gamma = gamma

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)  # Random action (explore)
        
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])  # Best action (exploit)
        
    def train(self, states, actions, rewards, next_states, dones):

        target_q = self.target_model.predict(next_states, verbose=0)
        max_q_next = np.max(target_q, axis=1)
        target_values = rewards + (1 - dones) * self.gamma * max_q_next

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)

            indices = np.arange(q_values.shape[0])
            selected_q = tf.gather(q_values, actions[:, None], batch_dims=1)
            loss = self.loss_fn(target_values, selected_q[:, 0])

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())



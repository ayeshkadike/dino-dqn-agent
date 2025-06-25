import tensorflow as tf
import numpy as np

class DQNAgent:
    def __init__(self, model, target_model, num_actions, gamma=0.99, lr=1e-4):
        self.model = model
        self.target_model = target_model
        self.num_actions = num_actions
        self.gamma = gamma

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.loss_fn = tf.keras.losses.Huber()

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]

        next_q_values = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_values, axis=1)

        target_q_values = self.target_model.predict(next_states, verbose=0)
        target_values = rewards + (1 - dones) * self.gamma * target_q_values[np.arange(batch_size), next_actions]

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            selected_q_values = tf.gather(q_values, actions[:, None], batch_dims=1)
            loss = self.loss_fn(target_values, tf.squeeze(selected_q_values, axis=1))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

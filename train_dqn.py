import numpy as np
import os
import re
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from dino_env import DinoEnv
from replay_buffer import ReplayBuffer
from model import build_dqn_model
from DQNAgent import DQNAgent

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[✔] GPU detected: {gpus[0].name}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"[!] GPU setup error: {e}")
else:
    print("[✘] No GPU detected. Training will use CPU.")



# directory for saved models
os.makedirs("checkpoints", exist_ok=True)

# Hyperparameters
num_episodes = 6100
batch_size = 64
epsilon = 0.0
epsilon_min = 0.1
epsilon_decay = 0.992
buffer_warmup     = 500
frame_skip = 4
target_sync_steps = 1000
global_step = 0
gamma = 0.99
lr = 1e-4

# Load latest checkpoint if available
def load_latest_checkpoint(model, target_model):
    latest_episode = 0
    latest_file = None

    for filename in os.listdir("checkpoints"):
        match = re.match(r"dqn_episode_(\d+)\.h5", filename)
        if match:
            episode_num = int(match.group(1))
            if episode_num > latest_episode:
                latest_episode = episode_num
                latest_file = filename

    if latest_file:
        path = os.path.join("checkpoints", latest_file)
        print(f"[✔] Resuming from checkpoint: {path}")
        model.load_weights(path)
        target_model.load_weights(path)

    return latest_episode

env = DinoEnv()
buffer = ReplayBuffer(max_size=50000, input_shape=(84, 84, 4))
model = build_dqn_model(input_shape=(84, 84, 4), num_actions=3)
target_model = build_dqn_model(input_shape=(84, 84, 4), num_actions=3)
target_model.set_weights(model.get_weights())
agent = DQNAgent(model, target_model, num_actions=3, gamma=gamma, lr=lr)

start_episode = load_latest_checkpoint(model, target_model)

# Load or warm up replay buffer
if os.path.exists("checkpoints/replay_buffer.pkl"):
    print(f"[✔] Loading replay buffer from checkpoints/replay_buffer.pkl")
    with open("checkpoints/replay_buffer.pkl", "rb") as f:
        buffer = pickle.load(f)
else:
    print("[•] Warming up replay buffer using trained model...")
    while buffer.size() < buffer_warmup:
        state = env.reset()
        done = False
        while not done and buffer.size() < buffer_warmup:
            action = agent.select_action(state, epsilon=0.2)
            next_state, reward, done, _ = env.step(action)
            buffer.store(state, action, reward, next_state, done)
            state = next_state
    with open("checkpoints/replay_buffer.pkl", "wb") as f:
        pickle.dump(buffer, f)
    print(f"[✔] Saved warmed-up buffer to checkpoints/replay_buffer.pkl")


episode_rewards = []
episode_losses = []
eval_scores = []

# Training loop
for episode in range(start_episode + 1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    total_loss = 0
    step = 0
    done = False

    while not done:

        # ε-greedy decision only on frames we actually step
        if global_step % frame_skip == 0:
            action = agent.select_action(state, epsilon)
            
        # else we repeat the last_action
        next_state, reward, done, _ = env.step(action)

        # buffer & bookkeeping
        buffer.store(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step        += 1
        global_step += 1

        # ----------------------train only after warm-up ---------------------------------
        if buffer.size() >= buffer_warmup:
            states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
            
            # GPU/CPU context for training
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                loss = agent.train(states, actions, rewards, next_states, dones)

            total_loss += loss

        # ----- hard-update target net every N env steps -----------------
        if global_step % target_sync_steps == 0:
            agent.update_target_model()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_min)

    episode_rewards.append(total_reward)
    avg_loss = total_loss / step if (step and total_loss) else 0
    episode_losses.append(avg_loss)

    print(f"Ep {episode:03d} | Reward: {total_reward:.0f} | Steps: {step} | "
          f"Epsilon: {epsilon:.3f} | Avg Loss: {avg_loss:.5f}")

    # Eval run
    if episode % 20 == 0:
        eval_state = env.reset()
        eval_done = False
        eval_score = 0
        action = 0
        while not eval_done:
            action = agent.select_action(eval_state, epsilon=0)
            eval_state, reward, eval_done, _ = env.step(action)
            eval_score += reward
        eval_scores.append((episode, eval_score))
        print(f"[Eval] Episode {episode}: Score = {eval_score}")

    # Save model
    if episode % 100 == 0:
        model.save(f"checkpoints/dqn_episode_{episode}.h5")
        print(f"Model saved at episode {episode}")

# Plot performance
plt.plot(episode_rewards)
plt.title("Episode Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

plt.plot(episode_losses)
plt.title("Average Training Loss Over Time")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.show()
env.close()

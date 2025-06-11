import numpy as np
from dino_env import DinoEnv
from model import build_dqn_model
from DQNAgent import DQNAgent

# === Configuration ===
checkpoint_path = "checkpoints/dqn_episode_6000.h5"  # Update if needed
input_shape = (84, 84, 4)
num_actions = 3
epsilon = 0.0
num_eval_episodes = 20

# === Load environment and model ===
env = DinoEnv()
model = build_dqn_model(input_shape=input_shape, num_actions=num_actions)
model.load_weights(checkpoint_path)

# === Create agent with model only (no training) ===
agent = DQNAgent(model=model, target_model=model, num_actions=num_actions)

# === Run Evaluation Episodes ===
print(f"[✔] Loaded model from {checkpoint_path}")
print(f"[ℹ] Starting evaluation over {num_eval_episodes} episodes...")

for episode in range(1, num_eval_episodes + 1):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = agent.select_action(state, epsilon=epsilon)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

    print(f"[Eval {episode:02d}] Reward: {total_reward:.2f} | Steps: {steps}")

env.close()

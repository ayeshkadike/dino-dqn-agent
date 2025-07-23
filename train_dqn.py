# ---------- Standard libs ----------
import os, re, pickle, random
from pathlib import Path
# ---------- Third-party ----------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# ---------- Local modules ----------
from dino_env      import DinoEnv, RuleBasedAgent
from replay_buffer import ReplayBuffer
from model         import build_dqn_model          # dueling DQN
from DQNAgent      import DQNAgent                # double-DQN + Huber

# ───────────────────────────────────────────────────────────────────────────────
# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus[0].name}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"[!] GPU setup error: {e}")
else:
    print("No GPU detected → using CPU")

device = '/GPU:0' if gpus else '/CPU:0'
Path("checkpoints").mkdir(exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
num_episodes      = 23000
batch_size        = 64
buffer_warmup     = 3_000         # expert steps before any training
frame_skip        = 2
target_sync_steps = 1_000
gamma             = 0.99
lr                = 1e-4

# expert-mix schedule
expert_prob_start   = 0.30        # 30 %
expert_prob_final   = 0.00
expert_decay_eps    = 2_000       # linear decay for first 2 000 eps

# tiny exploration for DQN when expert not used
dqn_epsilon = 0.05

# ───────────────────────────────────────────────────────────────────────────────
# Utility: latest checkpoint loader
def load_latest_ckpt(model, target):
    latest_ep = 0
    for f in os.listdir("checkpoints"):
        m = re.match(r"dqn_episode_(\d+)\.h5", f)
        if m:
            ep = int(m.group(1))
            if ep > latest_ep:
                latest_ep, latest_file = ep, f
    if latest_ep:
        model.load_weights(f"checkpoints/{latest_file}")
        target.load_weights(f"checkpoints/{latest_file}")
        print(f"Resumed from {latest_file}")
    return latest_ep

# ───────────────────────────────────────────────────────────────────────────────
# Environment + models
env         = DinoEnv()
expert      = RuleBasedAgent(env.driver)
buffer      = ReplayBuffer(max_size=50_000, input_shape=(84,84,4))

model       = build_dqn_model(input_shape=(84,84,4), num_actions=3)
model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
              loss=tf.keras.losses.Huber())

target      = build_dqn_model(input_shape=(84,84,4), num_actions=3)
target.set_weights(model.get_weights())

agent       = DQNAgent(model, target, num_actions=3, gamma=gamma, lr=lr)
start_ep    = load_latest_ckpt(model, target)

# ───────────────────────────────────────────────────────────────────────────────
# Buffer warm-up with 100 % expert
if Path("checkpoints/replay_buffer.pkl").exists():
    with open("checkpoints/replay_buffer.pkl", "rb") as f:
        buffer = pickle.load(f)
    print("Replay buffer loaded")
else:
    print("Warm-up: collecting expert transitions …")
    while buffer.size() < buffer_warmup:
        s = env.reset(); done = False
        while not done and buffer.size() < buffer_warmup:
            a = expert.select_action()
            s2,r,done,_ = env.step(a)
            buffer.store(s,a,r,s2,done)
            s = s2
    with open("checkpoints/replay_buffer.pkl","wb") as f:
        pickle.dump(buffer,f)
    print(f"Warm-up complete ({buffer.size()} steps)")

# ───────────────────────────────────────────────────────────────────────────────
# Training loop
episode_rewards, episode_losses, eval_scores = [], [], []
global_step = 0

for ep in range(start_ep+1, num_episodes+1):

    # linear expert-prob decay
    expert_prob = max(
        expert_prob_final,
        expert_prob_start * (1 - min(ep-1, expert_decay_eps)/expert_decay_eps)
    )

    s = env.reset(); done=False
    tot_reward = tot_loss = steps = 0

    while not done:
        # choose actor
        if random.random() < expert_prob:
            a = expert.select_action()
        else:
            a = agent.select_action(s, dqn_epsilon)

        # frame-skip logic: repeat last a for skipped frames
        if global_step % frame_skip != 0:
            a = prev_a
        prev_a = a

        s2,r,done,_ = env.step(a)
        buffer.store(s,a,r,s2,done)
        s = s2

        tot_reward += r
        steps      += 1
        global_step+= 1

        # training
        if buffer.size() >= buffer_warmup:
            states, actions, rewards, nxt_states, dones = buffer.sample_batch(batch_size)
            with tf.device(device):
                loss = agent.train(states, actions, rewards, nxt_states, dones)
            tot_loss += loss

        if global_step % target_sync_steps == 0:
            agent.update_target_model()

    episode_rewards.append(tot_reward)
    episode_losses.append(tot_loss/steps if steps else 0)
    print(f"Ep {ep:04d} | Rwd {tot_reward:.1f} | Steps {steps} | "
          f"ExpProb {expert_prob:.2f} | Loss {tot_loss:.3f}")

    # evaluation every 20 eps
    if ep % 20 == 0:
        es = 0; st = env.reset(); d=False
        while not d:
            a = agent.select_action(st, epsilon=0)
            st, rw, d,_ = env.step(a)
            es += rw
        eval_scores.append((ep, es))
        print(f"[Eval] Ep {ep}: score {es:.1f}")

    # save model + buffer every 100 eps
    if ep % 100 == 0:
        model.save(f"checkpoints/dqn_episode_{ep}.h5")
        with open("checkpoints/replay_buffer.pkl","wb") as f:
            pickle.dump(buffer,f)
        print(f"Saved checkpoint at ep {ep}")

# ───────────────────────────────────────────────────────────────────────────────
# Plots
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(episode_rewards); plt.title("Episode rewards"); plt.xlabel("Ep"); plt.ylabel("Reward")

plt.subplot(1,2,2)
plt.plot(episode_losses);  plt.title("Avg loss per episode"); plt.xlabel("Ep"); plt.ylabel("Loss")
plt.tight_layout(); plt.show()

env.close()

import time
import random
import cv2
import numpy as np
import mss
from dino_env import DinoEnv

monitor = {"top": 250, "left": 125, "width": 700, "height": 150}  # Example: right half of 1920x1080

# --- Step 2: Environment Test ---
env = DinoEnv(monitor=monitor)
obs = env.reset()
print("Environment reset. Initial observation shape:", obs.shape)

for step in range(20):
    action = random.choice([0, 1, 2])  # Random action: 0=do nothing, 1=jump, 2=duck
    obs, reward, done, info = env.step(action)
    print(f"Step {step}: Action={action}, Reward={reward}, Done={done}")
    cv2.imshow("Observation", obs)
    cv2.waitKey(100)
    if done:
        print("Game over detected. Resetting environment.")
        obs = env.reset()
        time.sleep(1)

cv2.destroyAllWindows()
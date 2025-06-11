import numpy as np
import cv2, mss, time
import random
from collections import deque
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

class DinoEnv:
    def __init__(self, monitor=None, stack_size=4):
        self.monitor = monitor or {"top": 235, "left": 600, "width": 700, "height": 165}
        self.sct = mss.mss()
        self.stack_size = stack_size
        self.frames = deque(maxlen=self.stack_size)

        # Launch Chrome
        opts = Options()
        opts.add_argument("start-maximized")
        self.driver = webdriver.Chrome(options=opts)
        self.driver.get("https://chromedino.com/")
        time.sleep(2)

        # Disable scrolling
        self.driver.execute_script("document.body.style.overflow='hidden'")
        self.body = self.driver.find_element("tag name", "body")
        self.body.send_keys(Keys.SPACE)
        time.sleep(1)

    def reset(self):
        self.body.send_keys(Keys.SPACE)
        time.sleep(0.2)
        frame, _ = self._get_observation()

        # Reset frame stack with same frame
        self.frames = deque([frame] * self.stack_size, maxlen=self.stack_size)
        return np.stack(self.frames, axis=-1)

    def step(self, action: int):
        # 0 = noop, 1 = jump, 2 = duck
        if action == 1:
            self.body.send_keys(Keys.SPACE)
        elif action == 2 and not self._is_game_over():
            self.body.send_keys(Keys.ARROW_DOWN)

        time.sleep(0.05)

        frame, _ = self._get_observation()
        self.frames.append(frame)
        stacked_obs = np.stack(self.frames, axis=-1)

        done = self._is_game_over()
        reward = 0.1 if not done else -1.0  # reward shaping

        return stacked_obs, reward, done, {}

    def _get_observation(self):
        img = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized, gray

    def _is_game_over(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def close(self):
        self.driver.quit()

    def render(self):
        if len(self.frames) == self.stack_size:
            merged = np.hstack(self.frames)
            cv2.imshow("Stacked Frames", merged)
            cv2.waitKey(1)

# --- Run Test ---
if __name__ == "__main__":
    monitor = {"top": 235, "left": 600, "width": 700, "height": 165}
    env = DinoEnv(monitor=monitor)

    obs = env.reset()
    print("Reset done. Observation shape:", obs.shape) 

    step = 0
    while True:
        action = random.choice([0, 1, 2])
        obs, reward, done, _ = env.step(action)
        env.render()  # optional
        print(f"Step {step}: Action={action}, Reward={reward}, Done={done}")
        if done:
            print("Game over. Resetting...")
            obs = env.reset()
            time.sleep(1)
        step += 1

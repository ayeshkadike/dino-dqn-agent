import numpy as np
import cv2
import mss
import pyautogui
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

class DinoEnv:
    def __init__(self, monitor=None):
        self.monitor = monitor or {"top": 250, "left": 125, "width": 700, "height": 150}
        self.sct = mss.mss()

        # Setup Chrome (not headless)
        options = Options()
        options.add_argument("start-maximized")
        self.driver = webdriver.Chrome(options=options)
        self.driver.get("https://chromedino.com/")
        time.sleep(2)

        # Start the game
        self.driver.find_element("tag name", "body").send_keys(Keys.SPACE)
        time.sleep(1)

    def reset(self):
        self.driver.find_element("tag name", "body").send_keys(Keys.SPACE)
        time.sleep(0.2)
        obs, _ = self._get_observation()
        return obs

    def step(self, action):
        if action == 1:
            pyautogui.press("space")
        elif action == 2:
            pyautogui.keyDown("down")
            time.sleep(0.1)
            pyautogui.keyUp("down")
        time.sleep(0.05)

        obs, _ = self._get_observation()
        done = self._is_game_over()
        reward = 1.0 if not done else 0.0
        return obs, reward, done, {}

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

# --- Run Test ---
if __name__ == "__main__":
    monitor = {"top": 225, "left": 600, "width": 700, "height": 165}
    env = DinoEnv(monitor=monitor)

    obs = env.reset()
    print("Environment reset. Initial observation shape:", obs.shape)

    step = 0
    while True:
        action = random.choice([0, 1, 2])
        obs, reward, done, _ = env.step(action)
        print(f"Step {step}: Action={action}, Reward={reward}, Done={done}")
        if done:
            print("Game over detected. Resetting environment.")
            obs = env.reset()
            time.sleep(1)
        step += 1

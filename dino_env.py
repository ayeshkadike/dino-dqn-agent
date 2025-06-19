import time
import random
from collections import deque

import cv2
import mss
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


#   RULE-BASED AGENT
class RuleBasedAgent:
    """
    Simple heuristic controller:

        • Jump  → any cactus when close
        • Duck  → low-flying pterodactyl (y < 90) when close
        • No-op → otherwise
    """

    JS_SNIPPET = """
        const r = Runner.instance_;
        const o = r.horizon.obstacles[0] || null;
        return o ? {x:o.xPos, y:o.yPos, type:o.typeConfig.type, speed:r.currentSpeed} : null;
    """

    def __init__(self, driver):
        self.driver = driver

    def select_action(self) -> int:
        """
        Returns
        -------
        0 : no-op
        1 : jump
        2 : duck
        """
        try:
            data = self.driver.execute_script(self.JS_SNIPPET)
            if data is None:
                return 0

            x, y, typ, speed = data["x"], data["y"], data["type"], data["speed"]

            reaction_threshold = 60 + int(speed * 7)      # farther when faster

            if x < reaction_threshold:              
                if "PTERODACTYL" in typ:
                    return 2 if y < 90 else 0   
                return 1                        
            return 0
        except Exception:
            return 0  


#   ENVIRONMENT WRAPPER
class DinoEnv:
    """Screen-grab + keyboard wrapper for Chrome Dino."""

    def __init__(self, monitor=None, stack_size: int = 4):
        self.monitor = monitor or {"top": 235, "left": 600, "width": 700, "height": 165}
        self.sct        = mss.mss()
        self.stack_size = stack_size
        self.frames     = deque(maxlen=stack_size)

        opts = Options()
        opts.add_argument("start-maximized")
        self.driver = webdriver.Chrome(options=opts)
        self.driver.get("https://chromedino.com/")

        # focus tab so key presses register
        body = self.driver.find_element("tag name", "body")
        body.click()
        body.send_keys(Keys.SPACE)
        time.sleep(1)

        self.body          = body
        self.action_chain  = ActionChains(self.driver)
        self.down_key_held = False

        self.driver.execute_script("document.body.style.overflow='hidden'")

        self.cheat_agent = RuleBasedAgent(self.driver)

    def reset(self):
        """Restart game and return stacked observation."""
        self.body.send_keys(Keys.SPACE)
        time.sleep(0.2)
        frame, _ = self._capture_frame()
        self.frames = deque([frame] * self.stack_size, maxlen=self.stack_size)
        return np.stack(self.frames, axis=-1)

    def step(self, action: int):
        """0=no-op, 1=jump, 2=duck"""
        if action == 1:
            self.body.send_keys(Keys.SPACE)
            self._release_duck()
        elif action == 2 and not self._is_game_over():
            if not self.down_key_held:
                self.action_chain.key_down(Keys.ARROW_DOWN).perform()
                self.down_key_held = True
        else:
            self._release_duck()

        time.sleep(0.05)

        frame, _ = self._capture_frame()
        self.frames.append(frame)
        stacked = np.stack(self.frames, axis=-1)

        done   = self._is_game_over()
        reward = 0.1 if not done else -1.0
        return stacked, reward, done, {}

    def cheat_step(self):
        """Use rule-based agent to choose action, then env.step()."""
        return self.step(self.cheat_agent.select_action())

    def render(self):
        if len(self.frames) == self.stack_size:
            cv2.imshow("Dino frames", np.hstack(self.frames))
            cv2.waitKey(1)

    def close(self):
        self.driver.quit()

    #   INTERNAL HELPERS
    def _release_duck(self):
        if self.down_key_held:
            self.action_chain.key_up(Keys.ARROW_DOWN).perform()
            self.down_key_held = False

    def _capture_frame(self):
        img = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0, gray

    def _is_game_over(self):
        return self.driver.execute_script("return Runner.instance_.crashed")


# Quick manual test
if __name__ == "__main__":
    env = DinoEnv()
    obs = env.reset()
    print("Reset ✓  | obs shape:", obs.shape)

    step = 0
    while True:
        obs, r, done, _ = env.cheat_step()
        env.render()
        print(f"step {step:04d} | r={r}")
        if done:
            print("Game over – resetting")
            obs = env.reset()
            time.sleep(1)
        step += 1

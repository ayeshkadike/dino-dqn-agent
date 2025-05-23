# dino-dqn-agent

A reinforcement learning agent trained using Deep Q-Networks (DQN) to play the Chrome Dino game using visual input, built with TensorFlow.

## Dependencies
- Tensorflow
- opencv-python
- pyautogui
- mss
- numpy
- selenium
- matplotlib

## How it works
- The game is launched in Chrome via Selenium (`https://chromedino.com`).
- The agent captures the game screen using `mss` and preprocesses it with OpenCV.
- Actions (jump/duck) are simulated using `pyautogui`.
- A convolutional DQN model is trained using TensorFlow/Keras.
- The agent uses experience replay, a target network, epsilon-greedy exploration, and evaluation checkpoints.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the training script: `python train_dqn.py`

The script will:

- Launch Chrome Dino in a visible window
- Begin training for the configured number of episodes
- Save a checkpoint every 50 episodes (checkpoints/dqn_episode_XX.h5)
- Resume automatically if rerun after interruption
- Show reward and loss plots when training finishes

You can customize hyper-parameters at the top of train_dqn.py.

---

**Note:** Make sure the Chrome Dino game is open and the game window is in a fixed position for consistent screen capture. 


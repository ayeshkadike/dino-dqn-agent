# dino-dqn-agent

A reinforcement learning agent trained using Deep Q-Networks (DQN) to play the Chrome Dino game using visual input, built with TensorFlow.

## Dependencies
- opencv-python
- pyautogui
- mss
- numpy
- torch
- Pillow
- Tensorflow

## How it works
- The agent captures the game screen using `mss` and processes it with OpenCV.
- Actions are sent to the game using `pyautogui` to simulate key presses.
- The RL model (DQN) is implemented in PyTorch.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`

---

**Note:** Make sure the Chrome Dino game is open and the game window is in a fixed position for consistent screen capture. 

## Project Structure
```
dino-rl/
│
├── main.py            # Main script to run the agent
├── dino_env.py        # Environment wrapper for the Dino game
├── model.py           # RL model (e.g., DQN)
├── utils.py           # Utility functions (screen capture, preprocessing, etc.)
├── requirements.txt   # Dependencies
└── README.md
``` 
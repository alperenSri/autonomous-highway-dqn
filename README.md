# 🏎️ Autonomous Highway Driving with Deep Q-Learning

## Members
Alperen Sari - 2103320 
Enes Burak Çetin - 2104091

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Library](https://img.shields.io/badge/Gymnasium-Highway--Env-green?style=for-the-badge)
![Algorithm](https://img.shields.io/badge/Algorithm-DQN%20(Stable--Baselines3)-orange?style=for-the-badge)

## 📌 Project Overview
This project trains an autonomous driving agent to navigate high-speed traffic using Reinforcement Learning. The goal is to maximize speed while avoiding collisions in a dynamic environment using the `highway-fast-v0` environment.

### 🎥 The Evolution (Visual Proof)
Below is the progression of the agent from random actions to fully autonomous driving.

* **Left:** Untrained Agent (Random/Fail)
* **Center:** Half-Trained Agent (Safe but Slow)
* **Right:** Fully Trained Agent (High Speed & Overtaking)

-----
Video
https://github.com/user-attachments/assets/08a3bbd0-c82a-481b-9269-3dd3b3600534
-----

## 📐 Methodology

### 1. The Math: Reward Function
To solve the multi-objective optimization problem (Speed vs. Safety), We utilized a composite reward function $R(s,a)$. The agent maximizes the expected return by balancing velocity and collision avoidance.

The reward function is defined as:

$$R(s,a) = \alpha \cdot \frac{v - v_{min}}{v_{max} - v_{min}} - \beta \cdot \mathbb{1}_{collision} + \gamma \cdot \text{lane\_keeping}$$

Where:
* $v$: Current velocity of the agent.
* $\alpha$: Weight for high speed (Normalized between $v_{min}$ and $v_{max}$).
* $\beta$: Heavy penalty for collisions (Terminal state).
* $\gamma$: Small reward for staying in the right-most lanes (to encourage traffic rules).

### 2. The Model: DQN & Architecture
I used **Deep Q-Network (DQN)** implemented via `Stable-Baselines3`. DQN was chosen for its stability in discrete action spaces like highway lane changing.

**Hyperparameters:**
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Algorithm** | DQN | Deep Q-Network |
| **Policy** | MlpPolicy | Multi-Layer Perceptron (Vector Input) |
| **Learning Rate** | `5e-4` | Optimized for stable convergence |
| **Batch Size** | `32` | Number of samples per gradient update |
| **Buffer Size** | `15,000` | Experience Replay memory capacity |
| **Gamma ($\gamma$)** | `0.8` | Discount factor (Prioritizes immediate survival) |
| **Network Arch** | `[256, 256]` | Two hidden layers with 256 neurons each |

---

## 📊 Training Analysis

### Reward vs. Episodes
The following graph demonstrates the learning curve over the training period.

-----
Training Graph
<img width="1000" height="500" alt="training_graph" src="https://github.com/user-attachments/assets/06c745ce-11a6-4463-be15-0b7b177de758" />
-----

### The Commentary
* **Phase 1 (Episodes 0-100):** The agent exhibits high exploration (Epsilon-greedy strategy). Rewards are low and volatile as the car frequently crashes while testing random actions.
* **Phase 2 (Episodes 100-300):** A sharp rise in the trend line. The agent learns that `collision = negative reward` and begins to prioritize staying in lanes. However, it often gets stuck behind slow cars.
* **Phase 3 (Convergence):** The agent masters the concept of "overtaking". It learns to change lanes to maintain $v_{max}$. The reward stabilizes at a high value, indicating consistent high-speed driving without crashing.
* **Emergency Brake**
Emergency Braking Behavior: While observing the fully trained agent, we notice distinct instances where the car decelerates rapidly (appearing to move backward relative to traffic flow). This is an emergent behavior where the agent learns to prioritize safety over speed in bottleneck situations. Instead of attempting risky overtakes that would lead to collisions, the agent chooses to brake, effectively minimizing the penalty function.

---

## 🛑 Challenges & Failures

### The "Looping" & Synchronization Hurdle
**The Problem:** During the video generation phase, We faced a significant technical challenge regarding the synchronization between the `Gymnasium` simulation speed and the video rendering FPS. Initially, the generated videos were either too short (ending immediately after a crash) or visually lagging ("slow-motion effect"), which failed to represent the true high-speed nature of the highway environment. Furthermore, when combining the videos, the "Untrained" agent's video would end in 2 seconds, causing the screen to go black while the other agents continued.

**The Solution:**
We engineered a custom video processing pipeline using `MoviePy`.
1.  **Freeze-Frame Logic:** We wrote a Python script to detect if a video ended prematurely (due to a crash). Instead of a black screen, the script extracts the final frame of the crash and extends it as a static image to match the duration of the fully trained agent.
2.  **Steps vs. Duration:** We modified the environment config to decouple the `duration` limit from the simulation steps, preventing the road lines from disappearing during long evaluation runs.

---

## 🛠️ Installation & Usage

```bash
# 1. Clone the repo
git clone [https://github.com/USERNAME/REPO_NAME.git](https://github.com/USERNAME/REPO_NAME.git)
cd REPO_NAME

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the agent
python train.py

# 4. Generate the Evolution Video

python evolution_video.py


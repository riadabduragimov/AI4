
# **Grid World Exploration**
**PersistentQLearner** class implements **reinforcement learning** using Q-learning:

1. **Initialization (`__init__`)**:
   - Defines **grid size** and possible **movement actions** (`N`, `E`, `W`, `S`).
   - Creates a **Q-table** that maps state-action pairs to rewards.
   - Sets **learning parameters**:
     - `learning_rate` : How much weight new information has.
     - `gamma` : Discount factor for future rewards.
     - `epsilon`: Exploration rate, controlling random vs. learned actions.
   - Initializes **experience replay** using a **deque** to store past experiences.

2. **Q-table Management**
   - Loads a previously stored Q-table or initializes a new one (`initialize_or_load_qtable`).
   - Saves the trained Q-table to a file (`save_qtable`).
   - Saves training history (`save_training_history`).

3. **State Prediction & Action Selection**
   - `predict_next_state()`: Determines the next state given an action.
   - `get_action()`: Uses an **epsilon-greedy strategy** (random vs. optimal action selection).
   - The agent favors **moving toward the goal** using Manhattan distance bias.

4. **Memory & Training**
   - Stores experiences using `remember()`.
   - Uses **experience replay** (`replay_experience`) to reinforce learning from stored experiences.
   - `train()`: Runs multiple episodes, interacting with the API (`reset_team`, `enter_world`, `make_move2`).
   - Adjusts exploration rate (`epsilon decay`).

5. **Training Workflow**
   - Initializes training & loads previous performance.
   - Iteratively selects **actions**, **collects rewards**, and **updates the Q-table**.
   - Saves progress periodically and upon achieving a new best reward.
   - Uses `api.make_move2()` for interaction, suggesting this is an external environment.

---

## Dependencies
Ensure you have the following installed:
```bash
pip install numpy
```

## Usage
Run the script to train the AI agent:
```bash
python epsilon.py
```

Modify training parameters as needed:
```python
agent.train(episodes=1000, max_steps=1000, save_interval=100)
```

## Files
- `qtable_worldX.pkl`: Stores trained Q-values.
- `training_historyX.pkl`: Tracks performance.
- `best_qtable_worldX_rewardY.pkl`: Stores best-performing Q-table.


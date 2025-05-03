from collections import deque, defaultdict
import random
import numpy as np
import pickle
import os
import api
from datetime import datetime

teamId = "1459"
worldId = "8"
otp = "5712768807"

class PersistentQLearner:
    def __init__(self, grid_size=40, gc=None):
        self.grid_size = grid_size
        self.goal_coords = gc 
        self.actions = ['N', 'E', 'W', 'S']  
        self.n_actions = len(self.actions)

        # File paths
        self.qtable_file = "qtable_world" + worldId + ".pkl"
        self.history_file = "training_history" + worldId + ".pkl"
        
        # Initialize or load Q-table
        self.q_table = self.initialize_or_load_qtable()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.best_reward = -np.inf
        self.training_history = []
        
        

    def initialize_or_load_qtable(self):
        """Initialize new Q-table or load existing one"""
        if os.path.exists(self.qtable_file):
            print("Loading existing Q-table...")
            with open(self.qtable_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("Initializing new Q-table...")
            return np.zeros((self.grid_size, self.grid_size, self.n_actions))

    def save_qtable(self):
        """Save Q-table to file"""
        with open(self.qtable_file, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {self.qtable_file}")

    def save_training_history(self):
        """Save training history to file"""
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'best_reward': self.best_reward, 
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.history_file, 'wb') as f:
            pickle.dump(history, f)
        print(f"Training history saved to {self.history_file}")

    
    def load_training_history(self):
        """Load training history from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'rb') as f:
                history = pickle.load(f)
            self.episode_rewards = history.get('episode_rewards', [])
            self.episode_steps = history.get('episode_steps', [])
            self.best_reward = history.get('best_reward', -np.inf)
            print(f"Loaded training history from {history['timestamp']}")
        else:
            print("No existing training history found")

    def predict_next_state(self, state, action):
        """Predict next state based on current state and action."""
        x, y = state
        if action == 0:   # 'N'
            return (x, min(y + 1, self.grid_size - 1))
        elif action == 1:  # 'E'
            return (min(x + 1, self.grid_size - 1), y)  
        elif action == 2:  # 'W'
            return (max(x - 1, 0), y)
        elif action == 3:  # 'S'
            return (x, max(y - 1, 0))
    
    def get_action(self, state, state_visits):
        """Epsilon-greedy action selection with goal-direction bias."""
        possible_actions = list(range(self.n_actions))
        valid_actions = []

        # Filter out over-visited states
        for action in possible_actions:
            next_state = self.predict_next_state(state, action)
            if state_visits.get(next_state, 0) < 2:
                valid_actions.append(action)

        if not valid_actions:
            valid_actions = possible_actions

        # Epsilon-greedy: Explore or Exploit
        if random.random() < self.epsilon:
            return random.choice(valid_actions)  
        else:
            # Bias toward actions moving closer to the goal
            if self.goal_coords:
                action_scores = []
                for action in valid_actions:
                    next_state = self.predict_next_state(state, action)
                    # Manhattan distance to goal
                    distance = abs(next_state[0] - self.goal_coords[0]) + abs(next_state[1] - self.goal_coords[1])
                    # Higher Q-value + lower distance = better action
                    action_score = self.q_table[state[0], state[1], action] - 0.1 * distance
                    action_scores.append(action_score)
                return valid_actions[np.argmax(action_scores)]
            else:
                # Fallback to standard Q-learning
                q_values = [self.q_table[state[0], state[1], a] for a in valid_actions]
                return valid_actions[np.argmax(q_values)]

    
    def remember(self, state, action, reward, next_state, done):
        if self.goal_coords and not done:
            # Calculate distance improvement
            old_dist = abs(state[0] - self.goal_coords[0]) + abs(state[1] - self.goal_coords[1])
            new_dist = abs(next_state[0] - self.goal_coords[0]) + abs(next_state[1] - self.goal_coords[1])
            reward += 0.5 * (old_dist - new_dist)  # Small bonus for moving closer
        self.memory.append((state, action, reward, next_state, done))

    
    
    def replay_experience(self):
        """Train on past experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.q_table[next_state[0], next_state[1]])
            
            current_q = self.q_table[state[0], state[1], action]
            self.q_table[state[0], state[1], action] += self.learning_rate * (target - current_q)

    
    
    def train(self, episodes = 1000, max_steps = 1000, save_interval = 100):
        """Main training loop"""
        self.load_training_history()
        
        for episode in range(episodes):
            api.reset_team(teamId, otp)
            api.enter_world(teamId, worldId)
            current_state = (0, 0)  # Start position
            previous_state = current_state
            total_reward = 0
            steps = 0
            done = False
            state_visits = defaultdict(int)
            state_visits[current_state] += 1
            
            while not done and steps < max_steps:
                action = self.get_action(current_state, state_visits)
                action_str = self.actions[action]
                # Get new state and reward from environment
                new_state, reward, done = api.make_move2(teamId, action_str, worldId)
                state_visits[new_state] += 1
                
                if reward <= -100:
                    current_state, action, reward, new_state, done = self.memory.pop()
                    reward = -1000
                    self.remember(current_state, action, reward, new_state, done)
                    break
                # Store experience
                self.remember(current_state, action, reward, new_state, done)
                total_reward += reward
                steps += 1
                previous_state = current_state
                current_state = new_state
                
            
            if done:
                self.goal_coords = previous_state
                self.epsilon = 0.1
                print("TARGET FOUND AT:", previous_state, total_reward)
            elif self.goal_coords != None:
                self.epsilon = 0.1
            else:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Experience replay
            self.replay_experience()
            
            # Decay exploration rate
            
            # Track performance
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.save_qtable()  # Save immediately when new best is found
                self.save_training_history()
                best_q_file = f"best_qtable_world{worldId}_reward{total_reward}.pkl"
                with open(best_q_file, 'wb') as f:
                    pickle.dump(self.q_table, f)
                print(f"New best reward! Saved Q-table to {best_q_file}")

                continue
            
            # Save progress periodically
            if episode % save_interval == 0:
                self.save_qtable()
                self.save_training_history()
                
                avg_reward = np.mean(self.episode_rewards[-save_interval:]) if self.episode_rewards else 0
                print(f"Episode {episode}, ε: {self.epsilon:.3f}, "
                      f"Reward: {total_reward}, Avg Reward: {avg_reward:.1f}, "
                      f"Steps: {steps}")
        
        # Final save
        self.save_qtable()
        self.save_training_history()
        print(f"Training complete! Best reward: {self.best_reward}")

 
    def get_optimal_path(self, filename, start=(0,0)):
        """Generate optimal path using Q-table loaded directly from file"""
        # Load Q-table from file
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Q-table file {self.qtable_file} not found")
        
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
        
        path = [start]
        current_state = start
        visited = set([start])
        api.reset_team(teamId, otp)
        api.enter_world(teamId, worldId)

        for _ in range(1600):  # Prevent infinite loops
            # Get action with highest Q-value from loaded table
            action = np.argmax(q_table[current_state[0], current_state[1]])
            action_str = self.actions[action]
            
            try:
                # Execute the move
                new_state, _, done = api.make_move2(teamId, action_str, worldId)
                
                    
                visited.add(new_state)
                path.append(new_state)
                current_state = new_state
                
                if done:
                    break
            except Exception as e:
                print(f"Error during path generation: {e}")
                break
                
        return path

# Usage example
if __name__ == "__main__":
    agent = PersistentQLearner(gc=(17, 11))
    
    agent.train(episodes=20, max_steps=1000, save_interval=1)
    
    # optimal_path = agent.get_optimal_path("best_qtable_world2_reward2030.pkl")
    # print(f"Optimal path length: {len(optimal_path)} steps")
    # print("Path sample:", optimal_path[:5], "...")
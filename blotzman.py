from collections import deque
import random
import numpy as np
import pickle
import os
import api
from datetime import datetime
import math

teamId = "1459"
worldId = "4"
otp = "5712768807"

class PersistentQLearner:
    def __init__(self, grid_size=40, exploration_strategy='epsilon_greedy'):
        self.grid_size = grid_size
        self.actions = ['N', 'E', 'W', 'S']  
        self.n_actions = len(self.actions)
        # File paths
        self.qtable_file = "qtable.pkl"
        self.history_file = "training_history.pkl"
        
        # Initialize or load Q-table
        self.q_table = self.initialize_or_load_qtable()
        
        # Learning parameters
        self.learning_rate = 0.2
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Boltzmann (Softmax) parameters
        self.temperature = 1.0
        self.temp_min = 0.1
        self.temp_decay = 0.995
        self.exploration_strategy = exploration_strategy
        
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

    
    def get_action(self, state):
        """Select action using current exploration strategy"""
        if self.exploration_strategy == 'e':
            return self.epsilon_greedy_action(state)
        elif self.exploration_strategy == 'b':
            return self.boltzmann_action(state)
        else:
            raise ValueError(f"Unknown exploration strategy: {self.exploration_strategy}")

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)  # Explore
        return np.argmax(self.q_table[state[0], state[1]])  # Exploit

    def boltzmann_action(self, state):
        q_values = self.q_table[state[0], state[1]]
        
        # Subtract max for numerical stability
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        
        # Sample action from the probability distribution
        return np.random.choice(range(self.n_actions), p=probabilities)

    def remember(self, state, action, reward, next_state, done):
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

    def train(self, episodes=1000, max_steps=1000, save_interval=100):
        self.load_training_history()
        
        for episode in range(episodes):
            current_state = (0, 0)  # Start position
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                action = self.get_action(current_state)
                action_str = self.actions[action]
                
                # Get new state and reward from environment
                new_state, reward, done = api.make_move2(teamId, action_str, worldId)
                
                # Store experience
                self.remember(current_state, action, reward, new_state, done)
                total_reward += reward
                steps += 1
                current_state = new_state
            
            # Experience replay
            self.replay_experience()
            
            # Decay exploration parameters
            if self.exploration_strategy == 'e':
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            elif self.exploration_strategy == 'b':
                self.temperature = max(self.temp_min, self.temperature * self.temp_decay)
            
            # Track performance
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
            
            # Save progress periodically
            if episode % save_interval == 0:
                self.save_qtable()
                self.save_training_history()
                
                avg_reward = np.mean(self.episode_rewards[-save_interval:]) if self.episode_rewards else 0
                print(f"Episode {episode}, "
                      f"Exploration: {self.epsilon if self.exploration_strategy == 'e' else self.temperature:.3f}, "
                      f"Reward: {total_reward}, Avg Reward: {avg_reward:.1f}, "
                      f"Steps: {steps}")
        
        # Final save
        self.save_qtable()
        self.save_training_history()
        print(f"Training complete! Best reward: {self.best_reward}")

    def get_optimal_path(self, start=(0,0)):
        """Generate optimal path using current Q-table"""
        path = [start]
        current_state = start
        visited = set([start])
        
        for _ in range(100):  # Prevent infinite loops
            action = np.argmax(self.q_table[current_state[0], current_state[1]])
            action_str = self.actions[action]
            
            new_state, _, done = api.make_move2(teamId, action_str, worldId)
            
            if new_state in visited:  # Prevent cycles
                break
                
            visited.add(new_state)
            path.append(new_state)
            current_state = new_state
            
            if done:
                break
                
        return path

# Usage example
if __name__ == "__main__":
    # Choose either 'epsilon_greedy => e' or 'boltzmann => b'
    agent = PersistentQLearner(exploration_strategy='b')
    
    agent.train(episodes=10, max_steps=500, save_interval=1)
    
    optimal_path = agent.get_optimal_path()
    print(f"Optimal path length: {len(optimal_path)} steps")
    print("Path sample:", optimal_path[:5], "...")
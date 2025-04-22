import random
import pickle
import os
from collections import deque
from gridworld_env import GridWorldEnv

class QLearning:
    def __init__(self, env: GridWorldEnv, world_id, alpha=0.2, gamma=0.95, epsilon=0.2, decay=0.995, replay_buffer_size=5000, batch_size=16):
        self.env = env
        self.world_id = world_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.Q = self.load_q_table(f"Q_world{world_id}.pkl")

    def load_q_table(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_q_table(self):
        with open(f"Q_world{self.world_id}.pkl", 'wb') as f:
            pickle.dump(self.Q, f)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.get_possible_actions())
        else:
            q_values = [self.Q.get((state, a), 0) for a in self.env.get_possible_actions()]
            max_val = max(q_values)
            actions = [a for a, v in zip(self.env.get_possible_actions(), q_values) if v == max_val]
            return random.choice(actions)

    def store_experience(self, state, action, reward, next_state, done):
        td_error = reward + self.gamma * max(self.Q.get((next_state, a), 0) for a in self.env.get_possible_actions()) - self.Q.get((state, action), 0)
        self.replay_buffer.append((state, action, reward, next_state, done, td_error))

    def sample_batch(self):
        sorted_buffer = sorted(self.replay_buffer, key=lambda x: abs(x[5]), reverse=True)
        return [(s, a, r, ns, d) for s, a, r, ns, d, _ in sorted_buffer[:self.batch_size]]

    def update_q_values(self, state, action, reward, next_state, done):
        next_action = self.select_action(next_state)
        q_next = self.Q.get((next_state, next_action), 0)
        self.Q[(state, action)] = self.Q.get((state, action), 0) + self.alpha * (reward + self.gamma * q_next - self.Q.get((state, action), 0))

    def train(self, episodes=1000):
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            if episode == 0:
                print("\nInitial Grid:")
                self.env.render()
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                self.update_q_values(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            if len(self.replay_buffer) > self.batch_size:
                for s, a, r, ns, d in self.sample_batch():
                    self.update_q_values(s, a, r, ns, d)

            self.epsilon *= self.decay
            rewards.append(total_reward)

            if episode == episodes - 1:
                print("\nFinal Grid after training:")
                self.env.render()

        self.save_q_table()
        return rewards

def extract_policy(Q, env: GridWorldEnv):
    policy = {}
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            state = (x, y)
            if state in env.terminal_states:
                continue
            q_values = {action: Q.get((state, action), 0) for action in env.get_possible_actions()}
            best_action = max(q_values, key=q_values.get)
            policy[state] = best_action
    return policy

def evaluate_agent(env: GridWorldEnv, policy, episodes=30):
    success_count, total_steps, total_rewards = 0, 0, 0
    for _ in range(episodes):
        state = env.reset()
        done, steps, total_reward = False, 0, 0
        while not done:
            action = policy.get(state, random.choice(env.get_possible_actions()))
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
        if state == env.goal_state:
            success_count += 1
        total_steps += steps
        total_rewards += total_reward

    print(f"Success Rate: {success_count / episodes * 100:.2f}%")
    print(f"Average Steps: {total_steps / episodes:.2f}")
    print(f"Average Reward: {total_rewards / episodes:.2f}")
    print("-" * 40)

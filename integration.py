import random
import pickle
import time
from api import enter_world, make_move, get_score, reset_team
import math

# Parameters
GRID_SIZE = 40
ACTIONS = ["N", "S", "E", "W"]
teamId = '1464'
otp = '5712768807' 
episodes = 5       
max_steps = 5000
alpha = 0.2         # Learning rate : you move 20% toward the new value and 80% stay with the old.
gamma = 0.95        # Discount factor : It tells the agent how much it should care about future rewards.

# Epsilon for exploration
initial_epsilon = 0.2 # It’s the starting value of epsilon at the very beginning of training.
min_epsilon = 0.05 # It’s the lowest value epsilon can decay down to.
decay_rate = 0.995 # It controls how fast the exploration rate (epsilon) shrinks over time.
epsilon = initial_epsilon

# Q-table: {(x, y): {action: value}}
Q = {(x, y): {a: 0.0 for a in ACTIONS} for x in range(GRID_SIZE) for y in range(GRID_SIZE)}

def choose_action(state, temperature):
    """
    Choose an action using Boltzmann exploration with a given temperature.
    Temperature is the key factor for exploration-exploitation tradeoff.
    """
    # Calculate the exponentiated Q-values
    exp_Q = {action: math.exp(Q[state][action] / temperature) for action in ACTIONS}
    total_exp = sum(exp_Q.values())

    # Probability distribution based on exponentiated Q-values (Softmax)
    probabilities = {action: exp_Q[action] / total_exp for action in exp_Q}

    # Choose an action based on the computed probabilities
    action = random.choices(ACTIONS, weights=[probabilities[action] for action in ACTIONS], k=1)[0]

    return action


def calculate_reward(state, next_state, reward, prev_reward):
    # If the current reward is higher than the previous reward, the agent is closer to the goal
    reward_gradient = reward - prev_reward  # Positive means moving towards goal
    reward_bonus = 0.1 if reward_gradient > 0 else 0  # Add a small bonus for moving towards the goal

    visit_penalty = -0.05 * (1 - math.exp(-abs(min(Q[state].values()))))  # Smarter penalty
    step_penalty = -0.01  # Small constant step penalty
    total_reward = reward + visit_penalty + step_penalty + reward_bonus
    return total_reward

def update_q_values(state, action, next_state, reward, prev_reward):
    reward_gradient = reward - prev_reward  # Positive means moving towards goal
    reward_bonus = 0.1 if reward_gradient > 0 else 0  # Reward bonus for increasing reward

    old_q = Q[state][action]
    future_q = max(Q[next_state].values())
    
    # Update Q-value considering the reward bonus
    Q[state][action] = old_q + alpha * (reward + reward_bonus + gamma * future_q - old_q)

def train_q_learning():
    global epsilon
    best_score = float('-inf')
    prev_reward = 0  # Initializing previous reward

    for ep in range(episodes):
        print(f"\nEpisode {ep+1} | Resetting team...")

        # Reset team with OTP before entering the world
        reset_team(teamId, otp)

        world_id = 1
        print(f"Entering World {world_id}...")

        try:
            start = enter_world(teamId, str(world_id))
        except AssertionError as e:
            print("Could not enter world:", e)
            break

        x, y = map(int, start.split(":"))
        state = (x, y)
        total_score = 0

        for step in range(max_steps):
            action = choose_action(state, epsilon)

            try:
                reward, new_state = make_move(teamId, action, world_id)
                new_x, new_y = int(new_state['x']), int(new_state['y'])
                next_state = (new_x, new_y)
            except Exception as e:
                print("Move failed:", e)
                break

            total_reward = calculate_reward(state, next_state, reward, prev_reward)
            update_q_values(state, action, next_state, total_reward, prev_reward)

            prev_reward = reward  # Update the previous reward

            state = next_state
            total_score += reward

        print(f"Episode {ep+1} finished | Total Score: {total_score:.2f} | ε: {epsilon:.2f}")
        epsilon = max(min_epsilon, epsilon * decay_rate)

        # Save best Q-table
        if total_score > best_score:
            best_score = total_score
            save_q_table("best_q_table.pkl")

        time.sleep(1)

    print("\nTraining complete.")

def save_q_table(path="q_table.pkl"):
    with open(path, "wb") as f:
        pickle.dump(Q, f)
    print(f"Q-table saved to {path}")

def load_q_table(path="q_table.pkl"):
    global Q
    with open(path, "rb") as f:
        Q = pickle.load(f)
    print(f"Q-table loaded from {path}")

if __name__ == "__main__":
    train_q_learning()
    save_q_table()

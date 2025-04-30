import random
import pickle
import time
import math
from api import enter_world, make_move, get_score, reset_team

# Parameters
GRID_SIZE = 40
ACTIONS = ["N", "S", "E", "W"]
teamId = '1459' # TESTRF2
otp = '5712768807' 

# Training parameters
episodes = 5       
max_steps = 5000
alpha = 0.2         # Learning rate
gamma = 0.95        # Discount factor
goal_reward = 100   # Expected reward when reaching goal

# Exploration parameters (Boltzmann)
initial_temperature = 1.0
min_temperature = 0.1
temperature_decay = 0.995
temperature = initial_temperature

# Initialize Q-table with small random values to break symmetry
Q = {(x, y): {a: random.uniform(-0.1, 0.1) for a in ACTIONS} 
     for x in range(GRID_SIZE) for y in range(GRID_SIZE)}

# Initialize visited traps set
visited_traps = set()

def choose_action(state):
    """
    Choose an action using Boltzmann exploration with current temperature.
    Higher temperature means more exploration.
    """
    # Handle unseen states (shouldn't happen with full initialization)
    if state not in Q:
        return random.choice(ACTIONS)
    
    # Calculate Boltzmann probabilities
    q_values = Q[state]
    max_q = max(q_values.values())
    
    # Subtract max for numerical stability
    exp_values = {a: math.exp((q - max_q) / max(temperature, 0.01)) 
                 for a, q in q_values.items()}
    total = sum(exp_values.values())
    
    # Normalize to probabilities
    probabilities = {a: v/total for a, v in exp_values.items()}
    
    # Choose action based on probabilities
    return random.choices(ACTIONS, weights=[probabilities[a] for a in ACTIONS])[0]

def calculate_reward(reward, prev_reward, steps, state):
    """
    Calculate modified reward with:
    - Reward gradient bonus (if improving)
    - Small step penalty
    - Large goal reward
    - Penalty for revisiting traps
    """
    # Basic reward components
    reward_gradient = reward - prev_reward
    step_penalty = -0.01 * steps  # Penalize longer paths
    
    # Bonus for moving toward goal
    direction_bonus = 0.1 if reward_gradient > 0 else 0
    
    # Penalty for revisiting traps
    trap_penalty = -30 if state in visited_traps else 0
    
    return reward + direction_bonus + step_penalty + trap_penalty

def update_q_values(state, action, next_state, reward):
    """
    Update Q-value using standard Q-learning update rule
    """
    # Handle unseen states
    if next_state not in Q:
        Q[next_state] = {a: 0.0 for a in ACTIONS}
    
    # Current Q-value
    current_q = Q[state][action]
    
    # Maximum Q-value for next state
    max_future_q = max(Q[next_state].values())
    
    # Q-learning update
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    Q[state][action] = new_q

def train_q_learning():
    global temperature
    best_score = float('-inf')
    training_stats = {
        'scores': [],
        'steps_to_goal': [],
        'exploration': []
    }

    for ep in range(episodes):
        print(f"\n=== Episode {ep+1}/{episodes} ===")
        print(f"Temperature: {temperature:.3f}")
        
        # Reset environment
        reset_team(teamId, otp)
        world_id = 1
        
        try:
            start = enter_world(teamId, str(world_id))
            x, y = map(int, start.split(":"))
            state = (x, y)
        except Exception as e:
            print(f"Failed to enter world: {e}")
            continue

        total_score = 0
        prev_reward = 0
        goal_reached = False

        for step in range(max_steps):
            action = choose_action(state)
            
            try:
                # Execute action
                reward, new_state = make_move(teamId, action, world_id)
                new_x, new_y = int(new_state['x']), int(new_state['y'])
                next_state = (new_x, new_y)
                
                # Calculate modified reward
                modified_reward = calculate_reward(reward, prev_reward, step, state)
                total_score += reward
                
                # Update Q-values
                update_q_values(state, action, next_state, modified_reward)
                
                # Check for goal
                if reward >= goal_reward:
                    print(f"Goal reached at step {step}!")
                    goal_reached = True
                    break
                
                # Update state
                prev_reward = reward
                state = next_state

                # Store visited trap state
                if reward < 0:  # Assume negative reward indicates a trap
                    visited_traps.add(state)
                
            except Exception as e:
                print(f"Move failed: {e}")
                break

        # Update training stats
        training_stats['scores'].append(total_score)
        training_stats['steps_to_goal'].append(step if goal_reached else max_steps)
        training_stats['exploration'].append(temperature)
        
        print(f"Episode {ep+1} results:")
        print(f"- Total score: {total_score:.2f}")
        print(f"- Steps: {step}")
        print(f"- Goal reached: {'Yes' if goal_reached else 'No'}")
        
        # Decay temperature
        temperature = max(min_temperature, temperature * temperature_decay)
        
        # Save best Q-table
        if total_score > best_score:
            best_score = total_score
            save_q_table("best_q_table2.pkl")
            print("New best Q-table saved!")
        
        time.sleep(1)  # Rate limiting

    print("\nTraining complete.")
    return training_stats

def save_q_table(path="q_table.pkl"):
    with open(path, "wb") as f:
        pickle.dump(Q, f)

def load_q_table(path="q_table.pkl"):
    global Q
    with open(path, "rb") as f:
        Q = pickle.load(f)
    print(f"Q-table loaded from {path}")
    print(Q)


if __name__ == "__main__":
    # Train the agent
    #load_q_table("best_q_table2.pkl")

    stats = train_q_learning()
    
    # Save final Q-table
    save_q_table()
    
    # Print some learned policy examples
    #print("\nSample learned policy:")
    #for state in random.sample(list(Q.keys()), min(5, len(Q))):
    #  best_action = max(Q[state].items(), key=lambda x: x[1])[0]
    #  print(f"State {state}: Best action = {best_action} (Q-value = {Q[state][best_action]:.2f})")

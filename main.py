from gridworld_env import GridWorldEnv
from q_learning import QLearning, extract_policy, evaluate_agent
import random

if __name__ == "__main__":
    GRID_SIZE = 40
    goal = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    env = GridWorldEnv(grid_size=GRID_SIZE, goal_state=goal, stochastic=True)
    agent = QLearning(env, world_id=0)
    rewards = agent.train(episodes=1500)
    policy = extract_policy(agent.Q, env)
    evaluate_agent(env, policy, episodes=10)

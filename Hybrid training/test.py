from environment import CreditLineEnv
from agent import QLearningAgent
from utils import plot_q_table_heatmap
import numpy as np

def test_agent(agent, env, n_episodes=100):
    """
    Test trained agent
    
    Args:
        agent: Trained QLearningAgent
        env: CreditLineEnv
        n_episodes: Number of test episodes
    """
    print("\n🧪 Testing Agent...")
    print("=" * 60)
    
    test_rewards = []
    test_lengths = []
    default_count = 0
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Use learned policy (no exploration)
            action = agent.choose_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done and next_state == 3:  # Default occurred
                default_count += 1
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
    
    # Print statistics
    print(f"\n📊 Test Results (over {n_episodes} episodes):")
    print("=" * 60)
    print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
    print(f"Default Rate: {default_count/n_episodes*100:.2f}%")
    print(f"Best Reward: {np.max(test_rewards):.2f}")
    print(f"Worst Reward: {np.min(test_rewards):.2f}")
    
    # Plot Q-table
    plot_q_table_heatmap(agent.q_table)
    
    return test_rewards, test_lengths

if __name__ == "__main__":
    # Load trained agent
    env = CreditLineEnv()
    agent = QLearningAgent()
    agent.load('trained_agent.pkl')
    
    # Test agent
    test_rewards, test_lengths = test_agent(agent, env, n_episodes=100)
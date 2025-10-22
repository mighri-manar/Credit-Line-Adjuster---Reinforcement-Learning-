"""
Offline Q-Learning Training with Real Home Credit Data
Implements the exact algorithm specification provided
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import HomeCreditDataLoader
from real_environment import RealDataCreditLineEnv
from agent import QLearningAgent
import pickle
import time

def offline_q_learning_train(max_episodes=10000, print_every=1000):
    """
    Train Q-Learning agent using real customer data following the exact algorithm
    
    Algorithm: Q-Learning for Credit Line Adjustment
    Input: 
    - States S = {S1: High Util/Poor History, S2: Moderate/Stable, S3: Low Util/Strong History, S4: Default}
    - Actions A = {A1: Increase, A2: Keep, A3: Decrease} -> Mapped to {2, 1, 0}
    - Learning rate α = 0.2
    - Discount factor γ = 0.95
    - Initial exploration rate ε = 0.2
    - Minimum exploration rate ε_min = 0.05
    - Decay rate = 0.995
    """
    
    print("="*80)
    print("OFFLINE Q-LEARNING TRAINING WITH HOME CREDIT DATASET")
    print("="*80)
    
    # Load real customer data
    print("\n1. Loading and preprocessing data...")
    data_loader = HomeCreditDataLoader()
    train_data, _ = data_loader.load_data()
    customer_episodes = data_loader.create_customer_episodes(train_data, max_episodes)
    
    # Create environment with real data
    print("\n2. Creating real data environment...")
    env = RealDataCreditLineEnv(data_loader=data_loader)
    
    # Display data statistics
    print("\n3. Dataset Statistics:")
    state_dist = env.get_state_distribution()
    for state_name, stats in state_dist.items():
        print(f"   {state_name}: {stats['count']} customers ({stats['percentage']:.1f}%)")
    
    print("\nDefault rates by state:")
    default_rates = env.get_default_rate_by_state()
    for state_name, stats in default_rates.items():
        print(f"   {state_name}: {stats['default_rate']:.1%} ({stats['defaults']}/{stats['total_customers']})")
    
    # Initialize Q-Learning Agent (following algorithm specification)
    print("\n4. Initializing Q-Learning Agent...")
    agent = QLearningAgent(
        n_states=4,
        n_actions=3,
        learning_rate=0.2,      # α = 0.2
        discount_factor=0.95,   # γ = 0.95
        epsilon=0.2,            # ε = 0.2
        epsilon_min=0.05,       # ε_min = 0.05
        epsilon_decay=0.995     # decay_rate = 0.995
    )
    
    print(f"   Learning rate (α): {agent.alpha}")
    print(f"   Discount factor (γ): {agent.gamma}")
    print(f"   Initial exploration rate (ε): {agent.epsilon}")
    print(f"   Minimum exploration rate (ε_min): {agent.epsilon_min}")
    print(f"   Decay rate: {agent.epsilon_decay}")
    
    # Training tracking
    episode_rewards = []
    episode_lengths = []
    state_action_counts = np.zeros((4, 3))
    total_rewards_by_state = np.zeros((4, 3))
    epsilon_history = []
    
    print(f"\n5. Starting Training on {max_episodes} customer episodes...")
    print("="*80)
    
    start_time = time.time()
    
    # Main training loop - following the algorithm exactly
    for episode in range(max_episodes):
        # Initialize customer state based on current profile
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Each customer episode (following algorithm structure)
        while True:
            # Epsilon-greedy action selection
            if np.random.random() < agent.epsilon:
                action = np.random.choice(3)  # Explore: random action
            else:
                action = np.argmax(agent.q_table[state])  # Exploit: best known action
            
            # Execute action: adjust customer's credit limit
            next_state, reward, done, truncated, step_info = env.step(action)
            
            # Track statistics
            episode_reward += reward
            episode_length += 1
            state_action_counts[state, action] += 1
            total_rewards_by_state[state, action] += reward
            
            # Q-Learning Update (Off-Policy TD Control)
            if next_state != 3:  # Non-terminal state (not default)
                best_next_Q = np.max(agent.q_table[next_state])
                TD_target = reward + agent.gamma * best_next_Q
                TD_error = TD_target - agent.q_table[state, action]
                agent.q_table[state, action] += agent.alpha * TD_error
            else:  # Terminal state (default)
                agent.q_table[state, action] += agent.alpha * (reward - agent.q_table[state, action])
                break  # End episode
            
            state = next_state  # Move to next state
            
            # Check for natural episode termination
            if done or truncated:
                break
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilon_history.append(agent.epsilon)
        
        # Decay exploration rate after each episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            
            print(f"Episode {episode + 1:6d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    training_time = time.time() - start_time
    
    print("="*80)
    print("TRAINING COMPLETED!")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Episodes per second: {max_episodes/training_time:.1f}")
    
    # Save the trained agent
    with open('trained_real_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    print("Trained agent saved to 'trained_real_agent.pkl'")
    
    # Calculate and display final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Final Q-table
    print("\nLearned Q-Table:")
    print("States: S1=High Risk, S2=Moderate, S3=Low Risk, S4=Default")
    print("Actions: 0=Decrease, 1=Keep, 2=Increase")
    print()
    print("       Decrease    Keep    Increase")
    state_names_short = ["S1", "S2", "S3", "S4"]
    for i in range(4):
        print(f"{state_names_short[i]:>3}: {agent.q_table[i, 0]:8.2f} {agent.q_table[i, 1]:8.2f} {agent.q_table[i, 2]:8.2f}")
    
    # Optimal policy
    print("\nOptimal Policy π*(s) = argmax_a Q(s,a):")
    action_names = ["Decrease", "Keep", "Increase"]
    for state in range(4):
        best_action = np.argmax(agent.q_table[state])
        print(f"  {env.state_names[state]}: {action_names[best_action]} (Q={agent.q_table[state, best_action]:.2f})")
    
    # Action frequency analysis
    print("\nAction Frequency by State:")
    for state in range(4):
        total_actions = np.sum(state_action_counts[state])
        if total_actions > 0:
            print(f"  {env.state_names[state]}:")
            for action in range(3):
                freq = state_action_counts[state, action] / total_actions
                avg_reward = total_rewards_by_state[state, action] / max(1, state_action_counts[state, action])
                print(f"    {action_names[action]}: {freq:.1%} (avg reward: {avg_reward:.2f})")
    
    # Performance metrics
    final_avg_reward = np.mean(episode_rewards[-1000:]) if len(episode_rewards) >= 1000 else np.mean(episode_rewards)
    total_reward = np.sum(episode_rewards)
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Episodes: {max_episodes}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Average Reward per Episode: {np.mean(episode_rewards):.2f}")
    print(f"  Final 1000 Episodes Avg Reward: {final_avg_reward:.2f}")
    print(f"  Standard Deviation: {np.std(episode_rewards):.2f}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    
    # Plot training progress
    plot_training_results(episode_rewards, epsilon_history, max_episodes)
    
    return agent, episode_rewards, state_action_counts

def plot_training_results(episode_rewards, epsilon_history, max_episodes):
    """Plot training progress"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Episode rewards with moving average
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue', linewidth=0.5)
    
    # Calculate moving average
    window = min(100, len(episode_rewards) // 10)
    if window > 1:
        moving_avg = pd.Series(episode_rewards).rolling(window).mean()
        plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
        plt.legend()
    
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Epsilon decay
    plt.subplot(1, 3, 2)
    plt.plot(epsilon_history, color='green')
    plt.title('Exploration Rate (ε) Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Reward distribution
    plt.subplot(1, 3, 3)
    plt.hist(episode_rewards, bins=50, alpha=0.7, color='purple')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_data_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training progress plots saved to 'real_data_training_progress.png'")

def test_trained_agent(agent_path='trained_real_agent.pkl', test_episodes=1000):
    """Test the trained agent on fresh data"""
    print("\n" + "="*80)
    print("TESTING TRAINED AGENT")
    print("="*80)
    
    # Load trained agent
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
    
    # Create test environment
    data_loader = HomeCreditDataLoader()
    _, test_data = data_loader.load_data()
    test_episodes_data = data_loader.create_customer_episodes(test_data, test_episodes)
    
    # Override the customer episodes for testing
    test_env = RealDataCreditLineEnv(data_loader=data_loader)
    test_env.customer_episodes = test_episodes_data
    
    test_rewards = []
    policy_actions = np.zeros((4, 3))  # Track policy decisions
    
    print(f"Testing on {len(test_episodes_data)} test customers...")
    
    for episode in range(len(test_episodes_data)):
        state, _ = test_env.reset()
        
        # Use learned policy (no exploration)
        action = agent.choose_action(state, training=False)
        next_state, reward, done, _, _ = test_env.step(action)
        
        test_rewards.append(reward)
        policy_actions[state, action] += 1
    
    # Test results
    avg_test_reward = np.mean(test_rewards)
    print(f"\nTest Results:")
    print(f"  Average Test Reward: {avg_test_reward:.2f}")
    print(f"  Test Reward Std: {np.std(test_rewards):.2f}")
    print(f"  Total Test Reward: {np.sum(test_rewards):.2f}")
    
    print(f"\nPolicy Distribution on Test Data:")
    action_names = ["Decrease", "Keep", "Increase"]
    for state in range(4):
        total = np.sum(policy_actions[state])
        if total > 0:
            print(f"  State {state} ({test_env.state_names[state]}):")
            for action in range(3):
                freq = policy_actions[state, action] / total
                print(f"    {action_names[action]}: {freq:.1%}")
    
    return avg_test_reward, test_rewards

if __name__ == "__main__":
    # Run offline training
    print("Starting Offline Q-Learning Training with Home Credit Dataset...")
    
    # Train the agent
    trained_agent, rewards, action_counts = offline_q_learning_train(
        max_episodes=10000,  # Use 10k episodes for comprehensive training
        print_every=1000
    )
    
    # Test the trained agent
    test_trained_agent()
    
    print("\n" + "="*80)
    print("OFFLINE Q-LEARNING TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
"""
Offline Multi-Agent Q-Learning Training with Real Home Credit Data
Implements multi-agent system with specialized agents for different risk states
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import HomeCreditDataLoader
from real_environment import RealDataCreditLineEnv
from multi_agent import MultiQLearningAgent, AdaptiveMultiAgent
import pickle
import time

def offline_multi_agent_train(max_episodes=10000, print_every=1000, use_adaptive=True):
    """
    Train Multi-Agent Q-Learning system using real customer data
    
    Multi-Agent Architecture:
    - Agent 0: High Risk Specialist (S1)
    - Agent 1: Moderate Risk Specialist (S2) 
    - Agent 2: Low Risk Specialist (S3)
    - Agent 3: Default Handler (S4)
    """
    
    print("="*80)
    print("OFFLINE MULTI-AGENT Q-LEARNING TRAINING")
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
    
    # Initialize Multi-Agent System
    print(f"\n4. Initializing Multi-Agent Q-Learning System...")
    print(f"   Using {'Adaptive' if use_adaptive else 'Standard'} Multi-Agent System")
    
    if use_adaptive:
        multi_agent = AdaptiveMultiAgent(
            learning_rate=0.2,      # α = 0.2
            discount_factor=0.95,   # γ = 0.95
            epsilon=0.2,            # ε = 0.2
            epsilon_min=0.05,       # ε_min = 0.05
            epsilon_decay=0.995     # decay_rate = 0.995
        )
    else:
        multi_agent = MultiQLearningAgent(
            learning_rate=0.2,
            discount_factor=0.95,
            epsilon=0.2,
            epsilon_min=0.05,
            epsilon_decay=0.995
        )
    
    print(f"   Number of Specialized Agents: {multi_agent.n_agents}")
    for agent_id, agent_name in multi_agent.agent_names.items():
        print(f"     Agent {agent_id}: {agent_name}")
    
    # Training tracking
    episode_rewards = []
    episode_lengths = []
    state_action_counts = np.zeros((4, 3))
    total_rewards_by_state = np.zeros((4, 3))
    epsilon_history = []
    agent_action_counts = {i: np.zeros((4, 3)) for i in range(4)}
    
    print(f"\n5. Starting Multi-Agent Training on {max_episodes} customer episodes...")
    print("="*80)
    
    start_time = time.time()
    
    # Main multi-agent training loop
    for episode in range(max_episodes):
        # Initialize customer state
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Each customer episode
        while True:
            # Multi-agent action selection
            action = multi_agent.choose_action(state, training=True)
            
            # Execute action
            next_state, reward, done, truncated, step_info = env.step(action)
            
            # Track statistics
            episode_reward += reward
            episode_length += 1
            state_action_counts[state, action] += 1
            total_rewards_by_state[state, action] += reward
            agent_action_counts[state][state, action] += 1  # Track which agent handled which state
            
            # Multi-Agent Q-Learning Update
            td_errors = multi_agent.update(state, action, reward, next_state, done)
            
            state = next_state
            
            # Check for episode termination
            if done or truncated:
                break
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilon_history.append(multi_agent.agents[0].epsilon)  # Track epsilon from first agent
        
        # Decay exploration rate for all agents
        multi_agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            
            print(f"Episode {episode + 1:6d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {multi_agent.agents[0].epsilon:.4f}")
    
    training_time = time.time() - start_time
    
    print("="*80)
    print("MULTI-AGENT TRAINING COMPLETED!")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Episodes per second: {max_episodes/training_time:.1f}")
    
    # Save the trained multi-agent system
    system_type = "adaptive" if use_adaptive else "standard"
    filename = f'trained_multi_agent_{system_type}.pkl'
    multi_agent.save(filename)
    print(f"Trained multi-agent system saved to '{filename}'")
    
    # Display comprehensive results
    print_multi_agent_results(multi_agent, env, episode_rewards, agent_action_counts)
    
    # Plot training progress
    plot_multi_agent_training_results(episode_rewards, epsilon_history, multi_agent, max_episodes)
    
    return multi_agent, episode_rewards, state_action_counts

def print_multi_agent_results(multi_agent, env, episode_rewards, agent_action_counts):
    """Print comprehensive multi-agent results"""
    print("\n" + "="*80)
    print("MULTI-AGENT FINAL RESULTS")
    print("="*80)
    
    # Agent-specific Q-tables and policies
    q_tables = multi_agent.get_q_tables()
    
    print("\nSpecialized Agent Q-Tables:")
    action_names = ["Decrease", "Keep", "Increase"]
    state_names_short = ["S1", "S2", "S3", "S4"]
    
    for agent_id in range(multi_agent.n_agents):
        agent_name = multi_agent.agent_names[agent_id]
        print(f"\n{agent_name}:")
        print("       Decrease    Keep    Increase")
        q_table = q_tables[agent_name]
        for i in range(4):
            print(f"{state_names_short[i]:>3}: {q_table[i, 0]:8.2f} {q_table[i, 1]:8.2f} {q_table[i, 2]:8.2f}")
        
        # Best action for each state according to this agent
        print("  Optimal Policy:")
        for state in range(4):
            best_action = np.argmax(q_table[state])
            print(f"    {env.state_names[state]}: {action_names[best_action]} (Q={q_table[state, best_action]:.2f})")
    
    # Agent performance statistics
    multi_agent.print_agent_statistics()
    
    # Overall performance metrics
    final_avg_reward = np.mean(episode_rewards[-1000:]) if len(episode_rewards) >= 1000 else np.mean(episode_rewards)
    total_reward = np.sum(episode_rewards)
    
    print(f"\nOverall Multi-Agent Performance:")
    print(f"  Total Episodes: {len(episode_rewards)}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Average Reward per Episode: {np.mean(episode_rewards):.2f}")
    print(f"  Final 1000 Episodes Avg Reward: {final_avg_reward:.2f}")
    print(f"  Standard Deviation: {np.std(episode_rewards):.2f}")
    
    # Action frequency by specialized agent
    print(f"\nAction Distribution by Specialized Agent:")
    for agent_id in range(multi_agent.n_agents):
        agent_name = multi_agent.agent_names[agent_id]
        print(f"\n{agent_name} handled State {agent_id} ({env.state_names[agent_id]}):")
        
        agent_counts = agent_action_counts[agent_id]
        total_actions = np.sum(agent_counts[agent_id])  # Actions for the agent's specialized state
        
        if total_actions > 0:
            for action in range(3):
                freq = agent_counts[agent_id, action] / total_actions
                print(f"  {action_names[action]}: {freq:.1%}")

def plot_multi_agent_training_results(episode_rewards, epsilon_history, multi_agent, max_episodes):
    """Plot multi-agent training progress"""
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Episode rewards with moving average
    plt.subplot(2, 4, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue', linewidth=0.5)
    
    # Calculate moving average
    window = min(100, len(episode_rewards) // 10)
    if window > 1:
        moving_avg = pd.Series(episode_rewards).rolling(window).mean()
        plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
        plt.legend()
    
    plt.title('Multi-Agent Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Epsilon decay
    plt.subplot(2, 4, 2)
    plt.plot(epsilon_history, color='green')
    plt.title('Exploration Rate (ε) Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Reward distribution
    plt.subplot(2, 4, 3)
    plt.hist(episode_rewards, bins=50, alpha=0.7, color='purple')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Agent performance comparison
    plt.subplot(2, 4, 4)
    performance = multi_agent.get_agent_performance()
    agent_names = [name.split()[0] for name in performance.keys()]  # Shortened names
    avg_rewards = [perf['avg_reward'] for perf in performance.values()]
    
    bars = plt.bar(agent_names, avg_rewards, color=['red', 'orange', 'green', 'blue'])
    plt.title('Agent Performance Comparison')
    plt.xlabel('Specialized Agents')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom')
    
    # Plots 5-8: Individual agent Q-tables heatmaps
    q_tables = multi_agent.get_q_tables()
    state_names = ["S1", "S2", "S3", "S4"]
    action_names = ["Decrease", "Keep", "Increase"]
    
    for i, (agent_name, q_table) in enumerate(q_tables.items()):
        plt.subplot(2, 4, 5 + i)
        im = plt.imshow(q_table, cmap='RdYlBu_r', aspect='auto')
        plt.title(f'{agent_name.split()[0]} Agent Q-Table')
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.xticks(range(3), action_names, rotation=45)
        plt.yticks(range(4), state_names)
        
        # Add text annotations
        for j in range(4):
            for k in range(3):
                plt.text(k, j, f'{q_table[j, k]:.1f}', 
                        ha='center', va='center', color='white' if abs(q_table[j, k]) > 50 else 'black')
        
        plt.colorbar(im, shrink=0.8)
    
    plt.tight_layout()
    
    # Save plot
    system_type = "adaptive" if isinstance(multi_agent, AdaptiveMultiAgent) else "standard"
    filename = f'multi_agent_{system_type}_training_progress.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Multi-agent training progress plots saved to '{filename}'")

def test_multi_agent_system(agent_path, test_episodes=1000):
    """Test the trained multi-agent system on fresh data"""
    print("\n" + "="*80)
    print("TESTING TRAINED MULTI-AGENT SYSTEM")
    print("="*80)
    
    # Load trained multi-agent system
    from multi_agent import MultiQLearningAgent, AdaptiveMultiAgent
    
    # Create a new multi-agent system and load the saved data
    if "adaptive" in agent_path:
        multi_agent = AdaptiveMultiAgent()
    else:
        multi_agent = MultiQLearningAgent()
    
    multi_agent.load(agent_path)
    
    # Create test environment
    data_loader = HomeCreditDataLoader()
    _, test_data = data_loader.load_data()
    test_episodes_data = data_loader.create_customer_episodes(test_data, test_episodes)
    
    test_env = RealDataCreditLineEnv(data_loader=data_loader)
    test_env.customer_episodes = test_episodes_data
    
    test_rewards = []
    policy_actions = np.zeros((4, 3))
    agent_test_performance = {i: [] for i in range(4)}
    
    print(f"Testing on {len(test_episodes_data)} test customers...")
    
    for episode in range(len(test_episodes_data)):
        state, _ = test_env.reset()
        
        # Use multi-agent system (no exploration)
        action = multi_agent.choose_action(state, training=False)
        next_state, reward, done, _, _ = test_env.step(action)
        
        test_rewards.append(reward)
        policy_actions[state, action] += 1
        agent_test_performance[state].append(reward)
    
    # Test results
    avg_test_reward = np.mean(test_rewards)
    print(f"\nMulti-Agent Test Results:")
    print(f"  Average Test Reward: {avg_test_reward:.2f}")
    print(f"  Test Reward Std: {np.std(test_rewards):.2f}")
    print(f"  Total Test Reward: {np.sum(test_rewards):.2f}")
    
    # Agent-specific test performance
    print(f"\nAgent-Specific Test Performance:")
    for agent_id in range(4):
        if agent_test_performance[agent_id]:
            agent_avg = np.mean(agent_test_performance[agent_id])
            agent_count = len(agent_test_performance[agent_id])
            print(f"  {multi_agent.agent_names[agent_id]}: {agent_avg:.2f} avg reward ({agent_count} episodes)")
    
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

def compare_single_vs_multi_agent(max_episodes=5000):
    """Compare single agent vs multi-agent performance"""
    print("\n" + "="*80)
    print("COMPARING SINGLE AGENT VS MULTI-AGENT SYSTEMS")
    print("="*80)
    
    # Train single agent (existing system)
    print("\n1. Training Single Agent System...")
    from agent import QLearningAgent
    
    data_loader = HomeCreditDataLoader()
    train_data, _ = data_loader.load_data()
    customer_episodes = data_loader.create_customer_episodes(train_data, max_episodes)
    env = RealDataCreditLineEnv(data_loader=data_loader)
    
    single_agent = QLearningAgent(
        n_states=4, n_actions=3, learning_rate=0.2,
        discount_factor=0.95, epsilon=0.2, epsilon_min=0.05, epsilon_decay=0.995
    )
    
    single_rewards = []
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            action = single_agent.choose_action(state, training=True)
            next_state, reward, done, truncated, _ = env.step(action)
            single_agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if done or truncated:
                break
        single_rewards.append(episode_reward)
        single_agent.decay_epsilon()
    
    # Train multi-agent system
    print("\n2. Training Multi-Agent System...")
    multi_agent, multi_rewards, _ = offline_multi_agent_train(max_episodes=max_episodes, print_every=max_episodes//5)
    
    # Compare results
    print("\n3. Performance Comparison:")
    print(f"Single Agent - Avg Reward: {np.mean(single_rewards):.2f} ± {np.std(single_rewards):.2f}")
    print(f"Multi-Agent  - Avg Reward: {np.mean(multi_rewards):.2f} ± {np.std(multi_rewards):.2f}")
    
    improvement = (np.mean(multi_rewards) - np.mean(single_rewards)) / abs(np.mean(single_rewards)) * 100
    print(f"Performance Improvement: {improvement:.1f}%")
    
    return single_rewards, multi_rewards

if __name__ == "__main__":
    print("Starting Multi-Agent Q-Learning Training with Home Credit Dataset...")
    
    # Train standard multi-agent system
    print("\n" + "="*50)
    print("TRAINING STANDARD MULTI-AGENT SYSTEM")
    print("="*50)
    trained_multi_agent, rewards, action_counts = offline_multi_agent_train(
        max_episodes=10000,
        print_every=1000,
        use_adaptive=False
    )
    
    # Test the trained system
    test_multi_agent_system('trained_multi_agent_standard.pkl')
    
    # Train adaptive multi-agent system
    print("\n" + "="*50)
    print("TRAINING ADAPTIVE MULTI-AGENT SYSTEM")
    print("="*50)
    trained_adaptive_agent, adaptive_rewards, adaptive_counts = offline_multi_agent_train(
        max_episodes=10000,
        print_every=1000,
        use_adaptive=True
    )
    
    # Test the adaptive system
    test_multi_agent_system('trained_multi_agent_adaptive.pkl')
    
    # Compare systems
    print("\n" + "="*50)
    print("SYSTEM COMPARISON")
    print("="*50)
    compare_single_vs_multi_agent(max_episodes=5000)
    
    print("\n" + "="*80)
    print("MULTI-AGENT Q-LEARNING TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
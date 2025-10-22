import numpy as np
from environment import CreditLineEnv
from agent import QLearningAgent
from utils import plot_training_progress

def train_agent(n_episodes=1000, print_every=100):
    """
    Train Q-Learning agent
    
    Args:
        n_episodes: Number of training episodes
        print_every: Print progress every N episodes
    """
    # Create environment and agent
    env = CreditLineEnv()
    agent = QLearningAgent(
        n_states=4,
        n_actions=3,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=0.2,
        epsilon_min=0.05,
        epsilon_decay=0.995
    )
    
    print("Starting Training...")
    print("=" * 60)
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update Q-table
            td_error = agent.update(state, action, reward, next_state, done)
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            
            # Move to next state
            state = next_state
        
        # Decay epsilon after episode
        agent.decay_epsilon()
        
        # Store episode metrics
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(agent.episode_rewards[-print_every:])
            avg_length = np.mean(agent.episode_lengths[-print_every:])
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-table range: [{agent.q_table.min():.2f}, {agent.q_table.max():.2f}]")
            print("-" * 60)
    
    print("\n✅ Training Complete!")
    
    # Display learned policy
    print("\n📊 Learned Policy:")
    print("=" * 60)
    policy = agent.get_policy()
    action_names = ["Decrease", "Keep", "Increase"]
    state_names = ["S1 (High Risk)", "S2 (Moderate)", "S3 (Low Risk)", "S4 (Default)"]
    
    for state in range(3):  # Exclude S4
        print(f"{state_names[state]:30} → {action_names[policy[state]]}")
    
    # Display Q-table
    print("\n📈 Final Q-Table:")
    print("=" * 60)
    print("State    | Decrease | Keep    | Increase")
    print("-" * 50)
    for state in range(4):
        q_values = agent.q_table[state]
        print(f"{state_names[state]:8} | {q_values[0]:8.2f} | {q_values[1]:7.2f} | {q_values[2]:8.2f}")
    
    # Save agent
    agent.save('trained_agent.pkl')
    
    # Plot training progress
    plot_training_progress(agent.episode_rewards, agent.episode_lengths)
    
    return agent, env

if __name__ == "__main__":
    agent, env = train_agent(n_episodes=1000, print_every=100)
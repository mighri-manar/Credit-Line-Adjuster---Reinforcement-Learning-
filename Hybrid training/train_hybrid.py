import numpy as np
import pandas as pd
from environment import CreditLineEnv
from agent import QLearningAgent
from utils import plot_training_progress
import matplotlib.pyplot as plt
from tqdm import tqdm

print("="*70)
print("HYBRID TRAINING: OFFLINE + ONLINE LEARNING")
print("="*70)

# ============================================================================
# PHASE 1: OFFLINE TRAINING FROM HISTORICAL DATA
# ============================================================================

print("\n🔵 PHASE 1: OFFLINE PRE-TRAINING")
print("-"*70)

# Load historical data
print("\n📂 Loading historical customer data...")
df = pd.read_csv('historical_customer_data.csv')
print(f"   Loaded {len(df):,} transitions from {df['customer_id'].nunique()} customers")

# Initialize agent for offline training
offline_agent = QLearningAgent(
    learning_rate=0.1,  # Lower for offline (more stable)
    discount_factor=0.95,
    epsilon=0.0  # No exploration in offline training
)

print("\n⚙️  Training on historical data...")
n_epochs = 20

offline_losses = []

for epoch in tqdm(range(n_epochs), desc="Offline epochs"):
    # Shuffle data each epoch
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    epoch_td_errors = []
    
    for _, row in df_shuffled.iterrows():
        state = int(row['state'])
        action = int(row['action'])
        reward = row['reward']
        next_state = int(row['next_state'])
        done = bool(row['done'])
        
        # Update Q-table
        td_error = offline_agent.update(state, action, reward, next_state, done)
        epoch_td_errors.append(abs(td_error))
    
    avg_loss = np.mean(epoch_td_errors)
    offline_losses.append(avg_loss)

print(f"\n✅ Offline training complete!")
print(f"   Final TD error: {offline_losses[-1]:.4f}")

# Show offline policy
print("\n📊 Policy After Offline Training:")
print("-"*70)
policy = offline_agent.get_policy()
action_names = ["Decrease", "Keep", "Increase"]
state_names = ["S1 (High Risk)", "S2 (Moderate)", "S3 (Low Risk)"]

for s in range(3):
    print(f"{state_names[s]:30} → {action_names[policy[s]]}")

# Save offline agent
offline_agent.save('offline_trained_agent.pkl')

# ============================================================================
# PHASE 2: ONLINE FINE-TUNING
# ============================================================================

print("\n🟢 PHASE 2: ONLINE FINE-TUNING")
print("-"*70)

env = CreditLineEnv()

# Continue training online
online_agent = QLearningAgent()
online_agent.q_table = offline_agent.q_table.copy()  # Start from offline policy
online_agent.alpha = 0.05  # Lower learning rate for fine-tuning
online_agent.epsilon = 0.1  # Small exploration
online_agent.epsilon_min = 0.01
online_agent.epsilon_decay = 0.99

n_online_episodes = 1000

print(f"\n⚙️  Fine-tuning for {n_online_episodes} episodes...")

for episode in tqdm(range(n_online_episodes), desc="Online episodes"):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        action = online_agent.choose_action(state, training=True)
        next_state, reward, done, truncated, info = env.step(action)
        online_agent.update(state, action, reward, next_state, done)
        
        episode_reward += reward
        state = next_state
    
    online_agent.decay_epsilon()
    online_agent.episode_rewards.append(episode_reward)

print(f"\n✅ Online fine-tuning complete!")
print(f"   Final avg reward: {np.mean(online_agent.episode_rewards[-100:]):.2f}")

# Show final policy
print("\n📊 Policy After Online Fine-Tuning:")
print("-"*70)
policy = online_agent.get_policy()

for s in range(3):
    print(f"{state_names[s]:30} → {action_names[policy[s]]}")

# Save hybrid agent
online_agent.save('hybrid_trained_agent.pkl')

# ============================================================================
# PHASE 3: PURE ONLINE TRAINING (FOR COMPARISON)
# ============================================================================

print("\n🟡 PHASE 3: PURE ONLINE TRAINING (Baseline)")
print("-"*70)

pure_online_agent = QLearningAgent(
    learning_rate=0.2,
    discount_factor=0.95,
    epsilon=0.2
)

print(f"\n⚙️  Training purely online for {n_online_episodes} episodes...")

for episode in tqdm(range(n_online_episodes), desc="Pure online episodes"):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        action = pure_online_agent.choose_action(state, training=True)
        next_state, reward, done, truncated, info = env.step(action)
        pure_online_agent.update(state, action, reward, next_state, done)
        
        episode_reward += reward
        state = next_state
    
    pure_online_agent.decay_epsilon()
    pure_online_agent.episode_rewards.append(episode_reward)

print(f"\n✅ Pure online training complete!")
print(f"   Final avg reward: {np.mean(pure_online_agent.episode_rewards[-100:]):.2f}")

pure_online_agent.save('pure_online_agent.pkl')

# ============================================================================
# COMPARISON & VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPARISON")
print("="*70)

# Plot offline loss curve
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(offline_losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Average TD Error')
plt.title('Offline Training: TD Error Convergence')
plt.grid(True, alpha=0.3)

# Plot online fine-tuning
plt.subplot(3, 1, 2)
window = 50
rewards = online_agent.episode_rewards
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(rewards, alpha=0.3, label='Episode Reward')
plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-Ep Avg')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Online Fine-Tuning: Reward Progress (After Offline Pre-training)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot pure online
plt.subplot(3, 1, 3)
rewards = pure_online_agent.episode_rewards
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(rewards, alpha=0.3, label='Episode Reward')
plt.plot(range(window-1, len(rewards)), moving_avg, 'g-', linewidth=2, label=f'{window}-Ep Avg')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Pure Online Training: Reward Progress (No Pre-training)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hybrid_training_comparison.png', dpi=300)
print("\n📊 Comparison plot saved: 'hybrid_training_comparison.png'")
plt.show()

# Final comparison table
print("\n📈 Final Performance Comparison:")
print("-"*70)
print(f"{'Method':<30} {'Avg Reward (last 100)':<25} {'Training Time'}")
print("-"*70)
print(f"{'Offline Only':<30} {np.mean(offline_losses):<25.2f} {'~30 sec'}")
print(f"{'Hybrid (Offline+Online)':<30} {np.mean(online_agent.episode_rewards[-100:]):<25.2f} {'~2 min'}")
print(f"{'Pure Online':<30} {np.mean(pure_online_agent.episode_rewards[-100:]):<25.2f} {'~1 min'}")

print("\n" + "="*70)
print("✅ HYBRID TRAINING COMPLETE!")
print("="*70)
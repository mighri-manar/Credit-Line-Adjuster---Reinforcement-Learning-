from environment import CreditLineEnv
from agent import QLearningAgent
import numpy as np

print("="*60)
print("QUICK TRAINING TEST (100 episodes)")
print("="*60)

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

print("\n🎯 Goal: Learn optimal credit limit policy")
print("   - S1 (High Risk) → Decrease")
print("   - S2 (Moderate) → Increase")
print("   - S3 (Low Risk) → Increase")

print("\n⏱️  Training for 100 episodes...\n")

# Training loop
for episode in range(100):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        action = agent.choose_action(state, training=True)
        next_state, reward, done, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        
        episode_reward += reward
        state = next_state
    
    agent.decay_epsilon()
    agent.episode_rewards.append(episode_reward)
    
    # Print progress every 20 episodes
    if (episode + 1) % 20 == 0:
        avg_reward = np.mean(agent.episode_rewards[-20:])
        print(f"Episode {episode + 1}/100 | Avg Reward: {avg_reward:6.2f} | ε: {agent.epsilon:.3f}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Show learned policy
print("\n📊 Learned Policy After 100 Episodes:")
print("-"*60)
policy = agent.get_policy()
action_names = ["Decrease", "Keep", "Increase"]
state_names = ["S1 (High Risk)", "S2 (Moderate)", "S3 (Low Risk)", "S4 (Default)"]

for s in range(3):
    best_action = action_names[policy[s]]
    print(f"{state_names[s]:30} → {best_action}")

# Show Q-table
print("\n📈 Q-Table After 100 Episodes:")
print("-"*60)
print("State    | Decrease | Keep    | Increase")
print("-"*50)
for s in range(4):
    q_vals = agent.q_table[s]
    best = "★" if s < 3 else " "
    best_idx = policy[s] if s < 3 else 0
    
    row = f"{state_names[s]:8} |"
    for a in range(3):
        marker = " ★" if a == best_idx and s < 3 else ""
        row += f" {q_vals[a]:7.2f}{marker:2} |"
    print(row)

# Check if policy is correct
print("\n🎯 Policy Evaluation:")
print("-"*60)
expected = [2, 2, 2]  # [Increase, Increase, Increase] based on MDP
correct = 0
for s in range(3):
    is_correct = policy[s] == expected[s]
    status = "✅" if is_correct else "❌"
    if is_correct:
        correct += 1
    print(f"{state_names[s]:30} {status}")

print(f"\nPolicy Accuracy: {correct}/3 states correct")

if correct == 3:
    print("🎉 Perfect! Agent learned the optimal policy!")
elif correct >= 2:
    print("👍 Good! Most states correct. Train longer for perfection.")
else:
    print("⚠️  Needs more training. Run full training (1500 episodes).")

print("\n💡 Next Step: Run full training with 'python train.py'")
from environment import CreditLineEnv
from agent import QLearningAgent
import numpy as np

print("="*70)
print("POLICY COMPARISON: Learned vs Random vs Optimal")
print("="*70)

env = CreditLineEnv()

# Load trained agent
trained_agent = QLearningAgent()
trained_agent.load('trained_agent.pkl')

# Define policies
def random_policy(state):
    """Random action selection"""
    return np.random.randint(0, 3)

def trained_policy(state):
    """Learned Q-Learning policy"""
    return trained_agent.choose_action(state, training=False)

def expected_optimal_policy(state):
    """Expected optimal from MDP analysis"""
    if state == 0:  # S1: High Risk
        return 0  # Decrease
    elif state == 1:  # S2: Moderate
        return 2  # Increase
    elif state == 2:  # S3: Low Risk
        return 2  # Increase
    else:
        return 0  # Default (doesn't matter)

# Test each policy
def test_policy(policy_fn, policy_name, n_episodes=100):
    """Test a policy and return metrics"""
    rewards = []
    lengths = []
    defaults = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = policy_fn(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done and state == 3:
                defaults += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    return {
        'name': policy_name,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_length': np.mean(lengths),
        'default_rate': defaults / n_episodes * 100
    }

# Run comparison
print("\n🔄 Running 100 episodes for each policy...\n")

random_results = test_policy(random_policy, "Random Policy")
trained_results = test_policy(trained_policy, "Trained Q-Learning")
optimal_results = test_policy(expected_optimal_policy, "Expected Optimal")

# Display results
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

policies = [random_results, trained_results, optimal_results]

print(f"\n{'Policy':<25} {'Avg Reward':<15} {'Avg Length':<15} {'Default Rate'}")
print("-"*70)

for p in policies:
    print(f"{p['name']:<25} {p['avg_reward']:>7.2f} ± {p['std_reward']:>5.2f}   "
          f"{p['avg_length']:>6.2f} months    {p['default_rate']:>5.1f}%")

# Determine winner
best_policy = max(policies, key=lambda x: x['avg_reward'])
print("\n" + "="*70)
print(f"🏆 BEST POLICY: {best_policy['name']}")
print(f"   Average Reward: {best_policy['avg_reward']:.2f}")
print("="*70)

# Show learned vs expected policy
print("\n📊 Policy Actions Comparison:")
print("-"*70)
print(f"{'State':<30} {'Learned':<15} {'Expected Optimal':<20} {'Match'}")
print("-"*70)

action_names = ["Decrease", "Keep", "Increase"]
state_names = ["S1 (High Risk)", "S2 (Moderate)", "S3 (Low Risk)"]

learned_policy = trained_agent.get_policy()

for s in range(3):
    learned = action_names[learned_policy[s]]
    expected = action_names[expected_optimal_policy(s)]
    match = "✅" if learned == expected else "❌"
    print(f"{state_names[s]:<30} {learned:<15} {expected:<20} {match}")

print("\n✅ Analysis Complete!")
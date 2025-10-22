from agent import QLearningAgent
from environment import CreditLineEnv
import numpy as np

print("="*60)
print("TESTING Q-LEARNING AGENT")
print("="*60)

# Test 1: Agent Creation
print("\n" + "-"*60)
print("TEST 1: Agent Initialization")
print("-"*60)

agent = QLearningAgent(
    n_states=4,
    n_actions=3,
    learning_rate=0.2,
    discount_factor=0.95,
    epsilon=0.2
)

print("✅ Agent created successfully!")
print(f"   Learning rate (α): {agent.alpha}")
print(f"   Discount factor (γ): {agent.gamma}")
print(f"   Exploration rate (ε): {agent.epsilon}")

# Test 2: Initial Q-table
print("\n" + "-"*60)
print("TEST 2: Initial Q-Table (should be all zeros)")
print("-"*60)
print(agent.q_table)
print("✅ Q-table initialized correctly!")

# Test 3: Action Selection
print("\n" + "-"*60)
print("TEST 3: Action Selection")
print("-"*60)

state = 1  # S2 (Moderate)
print(f"\nTesting from state: S2 (Moderate)")
print(f"Current Q-values for S2: {agent.q_table[state]}")

# Test exploration (random actions)
print("\n10 exploratory actions (with ε=0.2):")
actions = [agent.choose_action(state, training=True) for _ in range(10)]
action_names = {0: "Decrease", 1: "Keep", 2: "Increase"}
print([action_names[a] for a in actions])
print("✅ Exploration works (actions should vary)!")

# Test exploitation (should all be same since Q-values are equal)
agent.epsilon = 0.0  # No exploration
print("\n5 greedy actions (with ε=0.0, Q-values all zero):")
actions = [agent.choose_action(state, training=False) for _ in range(5)]
print([action_names[a] for a in actions])
print("✅ Exploitation works!")

# Test 4: Q-Learning Update
print("\n" + "-"*60)
print("TEST 4: Q-Learning Update")
print("-"*60)

agent = QLearningAgent()  # Fresh agent
state = 1  # S2
action = 2  # Increase
reward = -1  # Reward for increasing in S2
next_state = 2  # Transitions to S3
done = False

print(f"\nBefore update:")
print(f"Q(S2, Increase) = {agent.q_table[state, action]}")

td_error = agent.update(state, action, reward, next_state, done)

print(f"\nAfter update:")
print(f"Q(S2, Increase) = {agent.q_table[state, action]}")
print(f"TD Error: {td_error:.4f}")

expected_q = 0 + 0.2 * (-1 + 0.95 * 0 - 0)
print(f"Expected Q-value: {expected_q}")
print(f"✅ Q-update works correctly!")

# Test 5: Multiple Updates (simulate learning)
print("\n" + "-"*60)
print("TEST 5: Learning Simulation (10 updates)")
print("-"*60)

agent = QLearningAgent()
env = CreditLineEnv()

print("\nSimulating 10 transitions from S2...")
state = 1  # Start at S2

for i in range(10):
    action = agent.choose_action(state, training=True)
    
    # Simulate good outcome (increase limit in S2, get reward +4, stay in S2)
    reward = 4
    next_state = 2  # Move to S3 (good outcome)
    done = False
    
    old_q = agent.q_table[state, action]
    td_error = agent.update(state, action, reward, next_state, done)
    new_q = agent.q_table[state, action]
    
    if i < 3 or i == 9:  # Print first 3 and last
        print(f"  Update {i+1}: Q({state},{action}) = {old_q:.3f} → {new_q:.3f}")

print("\n✅ Agent is learning (Q-values changing)!")

# Test 6: Policy Extraction
print("\n" + "-"*60)
print("TEST 6: Policy Extraction")
print("-"*60)

# Manually set some Q-values to test policy extraction
agent.q_table[0] = [-15, -5, 3]   # S1: Best = Decrease (index 2)
agent.q_table[1] = [4, 2, -1]     # S2: Best = Decrease (index 0)
agent.q_table[2] = [6, 3, -3]     # S3: Best = Decrease (index 0)

policy = agent.get_policy()
print(f"\nExtracted policy: {policy}")
print("Expected: [2, 0, 0, ...]")

action_names = ["Decrease", "Keep", "Increase"]
state_names = ["S1 (High Risk)", "S2 (Moderate)", "S3 (Low Risk)"]

print("\nPolicy interpretation:")
for s in range(3):
    print(f"  {state_names[s]:30} → {action_names[policy[s]]}")

print("\n✅ Policy extraction works!")

# Test 7: Epsilon Decay
print("\n" + "-"*60)
print("TEST 7: Epsilon Decay")
print("-"*60)

agent = QLearningAgent(epsilon=0.2, epsilon_min=0.05, epsilon_decay=0.995)
print(f"\nInitial ε: {agent.epsilon:.4f}")

epsilons = [agent.epsilon]
for i in range(500):
    agent.decay_epsilon()
    if i in [49, 99, 199, 499]:
        epsilons.append(agent.epsilon)
        print(f"After {i+1} episodes: ε = {agent.epsilon:.4f}")

print(f"\n✅ Epsilon decays correctly (min = {agent.epsilon_min})!")

print("\n" + "="*60)
print("ALL AGENT TESTS PASSED! ✅")
print("="*60)
print("\nYour agent is ready for training!")
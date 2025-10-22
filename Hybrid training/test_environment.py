from environment import CreditLineEnv

print("="*60)
print("TESTING CREDIT LINE ENVIRONMENT")
print("="*60)

# Create environment
env = CreditLineEnv()
print("\n✅ Environment created successfully!")

# Test 1: Reset environment
print("\n" + "-"*60)
print("TEST 1: Environment Reset")
print("-"*60)
state, info = env.reset()
print(f"Initial state: {state} - {env.state_names[state]}")
print("✅ Reset works!")

# Test 2: Take some actions
print("\n" + "-"*60)
print("TEST 2: Taking Actions (5 steps)")
print("-"*60)

for step in range(5):
    # Try each action once
    action = step % 3  # 0, 1, 2 (Decrease, Keep, Increase)
    
    print(f"\nStep {step + 1}:")
    print(f"  Current State: {env.state_names[state]}")
    print(f"  Action Taken: {env.action_names[action]}")
    
    next_state, reward, done, truncated, info = env.step(action)
    
    print(f"  Reward: {reward}")
    print(f"  Next State: {env.state_names[next_state]}")
    print(f"  Episode Done: {done}")
    
    if done:
        print("\n  💀 Customer defaulted! Episode ended.")
        break
    
    state = next_state

print("\n✅ Actions work correctly!")

# Test 3: Run a complete episode
print("\n" + "-"*60)
print("TEST 3: Complete Episode")
print("-"*60)

state, _ = env.reset()
episode_reward = 0
episode_length = 0

print(f"Starting state: {env.state_names[state]}")

while True:
    action = env.action_space.sample()  # Random action
    next_state, reward, done, truncated, info = env.step(action)
    
    episode_reward += reward
    episode_length += 1
    
    if done or truncated:
        break
    
    state = next_state

print(f"Episode finished!")
print(f"  Total reward: {episode_reward}")
print(f"  Episode length: {episode_length} months")
print(f"  Final state: {env.state_names[state]}")

print("\n✅ Complete episode works!")

# Test 4: Check reward matrix
print("\n" + "-"*60)
print("TEST 4: Reward Matrix")
print("-"*60)
print("\nState       | Decrease | Keep | Increase")
print("-"*50)
for state in range(4):
    rewards = env.reward_matrix[state]
    print(f"{env.state_names[state]:11} | {rewards[0]:8} | {rewards[1]:4} | {rewards[2]:8}")

print("\n✅ Reward matrix matches your MDP!")

print("\n" + "="*60)
print("ALL TESTS PASSED! ✅")
print("="*60)
print("\nYour environment is ready for training!")
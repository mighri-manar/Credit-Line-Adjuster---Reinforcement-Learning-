import numpy as np
import pandas as pd
from environment import CreditLineEnv
from tqdm import tqdm  # Progress bar (install: pip install tqdm)

print("="*70)
print("GENERATING SYNTHETIC HISTORICAL CUSTOMER DATA")
print("="*70)

def generate_customer_history(env, n_customers=1000, max_months=24):
    """
    Simulate historical bank decisions on customers
    This mimics what a real bank's database would look like
    """
    print(f"\n📊 Generating data for {n_customers} customers...")
    print("(This simulates years of bank operations)\n")
    
    data = []
    
    for customer_id in tqdm(range(n_customers), desc="Customers processed"):
        state, _ = env.reset()
        
        for month in range(max_months):
            # Simulate bank's historical decisions
            # In reality, banks might have used rule-based policies
            # We'll simulate a "decent but not optimal" policy
            
            if state == 0:  # S1: High risk
                # Bank was cautious: 70% decrease, 20% keep, 10% increase
                action = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            elif state == 1:  # S2: Moderate
                # Bank was balanced: 20% decrease, 50% keep, 30% increase
                action = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
            elif state == 2:  # S3: Low risk
                # Bank was growth-focused: 10% decrease, 30% keep, 60% increase
                action = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])
            else:  # S4: Default
                action = 0  # Doesn't matter
            
            # Execute action and observe outcome
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store this transition (like a database record)
            data.append({
                'customer_id': customer_id,
                'month': month,
                'state': state,
                'state_name': env.state_names[state],
                'action': action,
                'action_name': env.action_names[action],
                'reward': reward,
                'next_state': next_state,
                'next_state_name': env.state_names[next_state],
                'done': done or truncated
            })
            
            state = next_state
            
            if done or truncated:
                break
    
    df = pd.DataFrame(data)
    return df

# Generate data
env = CreditLineEnv()
df = generate_customer_history(env, n_customers=1000, max_months=24)

# Save to CSV
df.to_csv('historical_customer_data.csv', index=False)

# Display statistics
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)

print(f"\n📈 Data Overview:")
print(f"   Total transitions: {len(df):,}")
print(f"   Unique customers: {df['customer_id'].nunique():,}")
print(f"   Average months per customer: {len(df) / df['customer_id'].nunique():.1f}")

print(f"\n📊 State Distribution:")
for state in range(4):
    count = (df['state'] == state).sum()
    pct = count / len(df) * 100
    print(f"   {df['state_name'].iloc[0] if state < len(df) else 'S4'}: {count:,} ({pct:.1f}%)")

print(f"\n🎬 Action Distribution:")
action_names = ['Decrease', 'Keep', 'Increase']
for action in range(3):
    count = (df['action'] == action).sum()
    pct = count / len(df) * 100
    print(f"   {action_names[action]}: {count:,} ({pct:.1f}%)")

print(f"\n💰 Reward Statistics:")
print(f"   Mean reward: {df['reward'].mean():.2f}")
print(f"   Std reward: {df['reward'].std():.2f}")
print(f"   Total reward: {df['reward'].sum():.2f}")

print(f"\n💀 Default Statistics:")
defaults = df[df['done'] == True]['customer_id'].nunique()
default_rate = defaults / df['customer_id'].nunique() * 100
print(f"   Customers defaulted: {defaults}/{df['customer_id'].nunique()} ({default_rate:.1f}%)")

print(f"\n✅ Data saved to: 'historical_customer_data.csv'")
print("="*70)
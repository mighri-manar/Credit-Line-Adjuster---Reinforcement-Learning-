import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import os

class HomeCreditDataLoader:
    """
    Data loader for Home Credit Default Risk dataset
    Maps real customer data to MDP states for Q-Learning
    """
    
    def __init__(self, data_dir='home-credit-default-risk'):
        self.data_dir = data_dir
        self.application_train = None
        self.application_test = None
        self.customer_episodes = []
        
        # State mapping criteria
        self.state_names = {
            0: "S1 (High Risk - High Utilization/Poor History)",
            1: "S2 (Moderate Risk - Moderate/Stable)",
            2: "S3 (Low Risk - Low Utilization/Strong History)",
            3: "S4 (Default)"
        }
        
    def load_data(self):
        """Load the main application datasets"""
        print("Loading Home Credit dataset...")
        
        train_path = os.path.join(self.data_dir, 'application_train.csv')
        test_path = os.path.join(self.data_dir, 'application_test.csv')
        
        self.application_train = pd.read_csv(train_path)
        self.application_test = pd.read_csv(test_path)
        
        print(f"Training data: {self.application_train.shape}")
        print(f"Test data: {self.application_test.shape}")
        
        return self.application_train, self.application_test
    
    def calculate_credit_utilization(self, df):
        """Calculate credit utilization ratio"""
        # Create utilization ratio based on credit amount vs income
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        
        # Handle missing values
        df['CREDIT_INCOME_RATIO'] = df['CREDIT_INCOME_RATIO'].fillna(df['CREDIT_INCOME_RATIO'].median())
        
        return df
    
    def calculate_payment_history_score(self, df):
        """Calculate payment history score based on available features"""
        # Create a composite score using multiple indicators
        score = np.zeros(len(df))
        
        # Days employed (more stable = better)
        employed_score = np.where(df['DAYS_EMPLOYED'] < 0, 
                                 np.clip(-df['DAYS_EMPLOYED'] / 365, 0, 10), 0)
        
        # Age factor (older = more stable)
        age_score = np.clip(-df['DAYS_BIRTH'] / 365 / 10, 0, 5)
        
        # Income stability
        income_score = np.clip(df['AMT_INCOME_TOTAL'] / 100000, 0, 5)
        
        # External sources (credit bureau info)
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        ext_score = 0
        for col in ext_sources:
            if col in df.columns:
                ext_score += df[col].fillna(0.5) * 3
        
        # Combine scores
        total_score = employed_score + age_score + income_score + ext_score
        
        # Normalize to 0-10 scale
        df['PAYMENT_HISTORY_SCORE'] = np.clip(total_score, 0, 10)
        
        return df
    
    def map_to_states(self, df):
        """
        Map customer data to MDP states based on credit utilization and payment history
        
        State mapping:
        - S1 (High Risk): High utilization (>0.8) OR low payment history (<3)
        - S2 (Moderate): Medium utilization (0.4-0.8) AND medium payment history (3-6)
        - S3 (Low Risk): Low utilization (<0.4) AND high payment history (>6)
        - S4 (Default): TARGET = 1
        """
        df = self.calculate_credit_utilization(df)
        df = self.calculate_payment_history_score(df)
        
        # Initialize states
        states = np.zeros(len(df), dtype=int)
        
        # Define thresholds
        util_high = df['CREDIT_INCOME_RATIO'].quantile(0.75)  # Top 25%
        util_low = df['CREDIT_INCOME_RATIO'].quantile(0.33)   # Bottom 33%
        
        hist_high = df['PAYMENT_HISTORY_SCORE'].quantile(0.67)  # Top 33%
        hist_low = df['PAYMENT_HISTORY_SCORE'].quantile(0.33)   # Bottom 33%
        
        # Map to states
        for i in range(len(df)):
            util = df.iloc[i]['CREDIT_INCOME_RATIO']
            hist = df.iloc[i]['PAYMENT_HISTORY_SCORE']
            target = df.iloc[i]['TARGET'] if 'TARGET' in df.columns else 0
            
            if target == 1:
                states[i] = 3  # S4 - Default
            elif util > util_high or hist < hist_low:
                states[i] = 0  # S1 - High Risk
            elif util > util_low and hist_low <= hist <= hist_high:
                states[i] = 1  # S2 - Moderate Risk
            else:
                states[i] = 2  # S3 - Low Risk
        
        df['STATE'] = states
        
        return df
    
    def create_customer_episodes(self, df, max_episodes=10000):
        """
        Create customer episodes for training
        Each episode represents a customer's journey through different states
        """
        print(f"Creating customer episodes from {len(df)} customers...")
        
        # Map customers to states
        df = self.map_to_states(df)
        
        episodes = []
        
        # For each customer, create an episode
        for idx, row in df.head(max_episodes).iterrows():
            initial_state = row['STATE']
            target = row['TARGET'] if 'TARGET' in df.columns else 0
            
            # Create episode data
            episode = {
                'customer_id': row['SK_ID_CURR'],
                'initial_state': initial_state,
                'target': target,
                'credit_amount': row['AMT_CREDIT'],
                'income': row['AMT_INCOME_TOTAL'],
                'utilization': row['CREDIT_INCOME_RATIO'],
                'history_score': row['PAYMENT_HISTORY_SCORE']
            }
            
            episodes.append(episode)
        
        self.customer_episodes = episodes
        
        # Print state distribution
        state_counts = df.head(max_episodes)['STATE'].value_counts().sort_index()
        print("\nState distribution:")
        for state, count in state_counts.items():
            print(f"  {self.state_names[state]}: {count} customers ({count/len(episodes)*100:.1f}%)")
        
        print(f"\nCreated {len(episodes)} customer episodes")
        return episodes
    
    def get_episode_reward(self, initial_state, action, target):
        """
        Calculate episode reward based on action taken and actual outcome
        
        Args:
            initial_state: Customer's initial risk state
            action: Credit line action taken (0=Decrease, 1=Keep, 2=Increase)
            target: Actual outcome (0=No default, 1=Default)
        
        Returns:
            reward: Calculated reward
        """
        # Base reward matrix aligned with the algorithm specification
        base_rewards = {
            0: [3, -5, -15],    # S1: Decrease=+3, Keep=-5, Increase=-15
            1: [-1, 2, 4],      # S2: Decrease=-1, Keep=+2, Increase=+4
            2: [-3, 3, 6],      # S3: Decrease=-3, Keep=+3, Increase=+6
            3: [-20, -20, -20]  # S4: Default (terminal)
        }
        
        base_reward = base_rewards[initial_state][action]
        
        # Adjust reward based on actual outcome
        if target == 1:  # Customer defaulted
            # Penalize more if we increased limit for a customer who defaulted
            if action == 2:  # Increase
                actual_reward = base_reward - 25
            elif action == 1:  # Keep
                actual_reward = base_reward - 15
            else:  # Decrease
                actual_reward = base_reward + 5  # Less penalty for being conservative
        else:  # Customer didn't default
            # Reward good decisions
            if initial_state == 0 and action == 0:  # Correctly decreased for high risk
                actual_reward = base_reward + 5
            elif initial_state == 2 and action == 2:  # Correctly increased for low risk
                actual_reward = base_reward + 3
            else:
                actual_reward = base_reward
        
        return actual_reward
    
    def get_training_summary(self):
        """Get summary statistics of the training data"""
        if not self.customer_episodes:
            return "No episodes created yet. Run create_customer_episodes() first."
        
        df = pd.DataFrame(self.customer_episodes)
        
        summary = {
            'total_episodes': len(self.customer_episodes),
            'default_rate': df['target'].mean(),
            'state_distribution': df['initial_state'].value_counts().to_dict(),
            'avg_credit_amount': df['credit_amount'].mean(),
            'avg_income': df['income'].mean(),
            'avg_utilization': df['utilization'].mean(),
            'avg_history_score': df['history_score'].mean()
        }
        
        return summary
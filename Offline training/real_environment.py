import numpy as np
import gymnasium as gym
from gymnasium import spaces
from data_loader import HomeCreditDataLoader

class RealDataCreditLineEnv(gym.Env):
    """
    Real Data Environment for Credit Line Adjustment using Home Credit dataset
    Implements offline training with real customer episodes
    """
    
    def __init__(self, data_loader=None, max_episodes=10000):
        super(RealDataCreditLineEnv, self).__init__()
        
        # Define state space: 0=S1, 1=S2, 2=S3, 3=S4 (terminal)
        self.observation_space = spaces.Discrete(4)
        
        # Define action space: 0=Decrease, 1=Keep, 2=Increase
        self.action_space = spaces.Discrete(3)
        
        # State names for interpretability
        self.state_names = {
            0: "S1 (High Risk - High Utilization/Poor History)",
            1: "S2 (Moderate Risk - Moderate/Stable)",
            2: "S3 (Low Risk - Low Utilization/Strong History)",
            3: "S4 (Default)"
        }
        
        # Action names
        self.action_names = {
            0: "Decrease Limit",
            1: "Keep Limit", 
            2: "Increase Limit"
        }
        
        # Initialize data loader
        if data_loader is None:
            self.data_loader = HomeCreditDataLoader()
            self.data_loader.load_data()
            self.customer_episodes = self.data_loader.create_customer_episodes(
                self.data_loader.application_train, max_episodes
            )
        else:
            self.data_loader = data_loader
            self.customer_episodes = data_loader.customer_episodes
        
        # Current episode tracking
        self.current_episode_idx = 0
        self.current_customer = None
        self.episode_step = 0
        self.max_steps_per_episode = 1  # Each customer is one decision point
        
        print(f"Environment initialized with {len(self.customer_episodes)} real customer episodes")
        
    def reset(self, seed=None, options=None):
        """
        Reset environment to start a new customer episode
        Returns the customer's initial state
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get next customer episode
        if self.current_episode_idx >= len(self.customer_episodes):
            # Shuffle episodes if we've gone through all customers
            np.random.shuffle(self.customer_episodes)
            self.current_episode_idx = 0
        
        self.current_customer = self.customer_episodes[self.current_episode_idx]
        self.current_episode_idx += 1
        self.episode_step = 0
        
        initial_state = self.current_customer['initial_state']
        
        info = {
            'customer_id': self.current_customer['customer_id'],
            'credit_amount': self.current_customer['credit_amount'],
            'income': self.current_customer['income'],
            'utilization': self.current_customer['utilization'],
            'history_score': self.current_customer['history_score']
        }
        
        return initial_state, info
    
    def step(self, action):
        """
        Execute action for current customer and return results
        
        Args:
            action: Credit line adjustment action (0=Decrease, 1=Keep, 2=Increase)
            
        Returns:
            next_state: Next state (S4 if default, otherwise based on action)
            reward: Reward based on action and actual customer outcome
            done: True if episode is terminated
            truncated: True if episode is truncated (not used in our case)
            info: Additional information
        """
        if self.current_customer is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        self.episode_step += 1
        
        # Get customer's actual outcome
        initial_state = self.current_customer['initial_state']
        actual_default = self.current_customer['target']
        
        # Calculate reward using real outcome
        reward = self.data_loader.get_episode_reward(initial_state, action, actual_default)
        
        # Determine next state based on action and outcome
        if actual_default == 1:
            next_state = 3  # S4 - Default (terminal)
            done = True
        else:
            # Customer didn't default - determine new state based on action
            next_state = self._get_next_state(initial_state, action)
            done = True  # Each customer episode is one decision
        
        truncated = False  # Not used in our offline learning setup
        
        info = {
            'customer_id': self.current_customer['customer_id'],
            'action_taken': self.action_names[action],
            'actual_default': actual_default,
            'reward_breakdown': {
                'base_reward': reward,
                'initial_state': self.state_names[initial_state],
                'final_state': self.state_names[next_state]
            }
        }
        
        return next_state, reward, done, truncated, info
    
    def _get_next_state(self, current_state, action):
        """
        Determine next state based on current state and action
        For customers who didn't default
        """
        # State transition logic based on credit line adjustments
        if current_state == 0:  # S1 - High Risk
            if action == 0:  # Decrease - likely to improve to moderate
                return 1  # S2
            elif action == 1:  # Keep - might stay or worsen
                return np.random.choice([0, 1], p=[0.6, 0.4])
            else:  # Increase - likely to stay high risk
                return 0
                
        elif current_state == 1:  # S2 - Moderate Risk
            if action == 0:  # Decrease - might improve to low risk
                return np.random.choice([1, 2], p=[0.7, 0.3])
            elif action == 1:  # Keep - likely to stay moderate
                return 1
            else:  # Increase - might improve to low risk
                return np.random.choice([1, 2], p=[0.4, 0.6])
                
        elif current_state == 2:  # S3 - Low Risk
            if action == 0:  # Decrease - likely to stay low risk
                return 2
            elif action == 1:  # Keep - stay low risk
                return 2
            else:  # Increase - stay low risk (good customers)
                return 2
                
        return current_state  # Default case
    
    def get_state_distribution(self):
        """Get distribution of initial states in the dataset"""
        states = [episode['initial_state'] for episode in self.customer_episodes]
        unique, counts = np.unique(states, return_counts=True)
        
        distribution = {}
        for state, count in zip(unique, counts):
            distribution[self.state_names[state]] = {
                'count': count,
                'percentage': (count / len(states)) * 100
            }
        
        return distribution
    
    def get_default_rate_by_state(self):
        """Get default rate by initial state"""
        state_defaults = {}
        
        for state in range(4):
            state_episodes = [ep for ep in self.customer_episodes if ep['initial_state'] == state]
            if state_episodes:
                defaults = sum(ep['target'] for ep in state_episodes)
                total = len(state_episodes)
                default_rate = defaults / total
                
                state_defaults[self.state_names[state]] = {
                    'total_customers': total,
                    'defaults': defaults,
                    'default_rate': default_rate
                }
        
        return state_defaults
    
    def render(self, mode='human'):
        """Render current state information"""
        if self.current_customer:
            print(f"\nCustomer ID: {self.current_customer['customer_id']}")
            print(f"State: {self.state_names[self.current_customer['initial_state']]}")
            print(f"Credit Amount: ${self.current_customer['credit_amount']:,.2f}")
            print(f"Income: ${self.current_customer['income']:,.2f}")
            print(f"Utilization Ratio: {self.current_customer['utilization']:.2f}")
            print(f"History Score: {self.current_customer['history_score']:.2f}")
            print(f"Actual Outcome: {'Default' if self.current_customer['target'] else 'No Default'}")
    
    def close(self):
        """Clean up environment"""
        pass
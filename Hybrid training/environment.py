import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CreditLineEnv(gym.Env):
    """
    Custom Environment for Credit Line Adjustment
    Following your MDP specification
    """
    
    def __init__(self):
        super(CreditLineEnv, self).__init__()
        
        # Define state space: 0=S1, 1=S2, 2=S3, 3=S4 (terminal)
        self.observation_space = spaces.Discrete(4)
        
        # Define action space: 0=Decrease, 1=Keep, 2=Increase
        self.action_space = spaces.Discrete(3)
        
        # State names for interpretability
        self.state_names = {
            0: "S1 (High Utilization, Poor History)",
            1: "S2 (Moderate Utilization, Stable History)",
            2: "S3 (Low Utilization, Strong History)",
            3: "S4 (Default)"
        }
        
        # Action names
        self.action_names = {
            0: "Decrease Limit",
            1: "Keep Limit",
            2: "Increase Limit"
        }
        
        # Reward matrix: reward[state][action]
        self.reward_matrix = {
    0: [3, -5, -15],    # S1: Decrease=+3, Keep=-5, Increase=-15
    1: [-1, 2, 4],      # S2: Decrease=-1, Keep=+2, Increase=+4
    2: [-3, 3, 6],      # S3: Decrease=-3, Keep=+3, Increase=+6
    3: [-20, -20, -20]  # S4: Default (terminal)
     } 
        
        # Transition probabilities: P[current_state][action] -> next_state
        # These simulate realistic customer behavior
        self.transition_probs = self._define_transitions()
        
        # Current state
        self.current_state = 0
        self.steps = 0
        self.max_steps = 24  # Max 24 months per episode
        
    def _define_transitions(self):
        """
        Define state transition probabilities
        Format: {state: {action: {next_state: probability}}}
        """
        transitions = {
            # From S1 (High Risk)
            0: {
                0: {0: 0.4, 1: 0.4, 2: 0.1, 3: 0.1},  # Decrease: might improve or default
                1: {0: 0.5, 1: 0.2, 2: 0.05, 3: 0.25}, # Keep: high default risk
                2: {0: 0.3, 1: 0.2, 2: 0.1, 3: 0.4}    # Increase: very risky
            },
            # From S2 (Moderate)
            1: {
                0: {0: 0.3, 1: 0.5, 2: 0.15, 3: 0.05}, # Decrease: might worsen
                1: {0: 0.15, 1: 0.6, 2: 0.2, 3: 0.05}, # Keep: stable
                2: {0: 0.1, 1: 0.3, 2: 0.55, 3: 0.05}  # Increase: likely improve
            },
            # From S3 (Low Risk)
            2: {
                0: {0: 0.2, 1: 0.5, 2: 0.25, 3: 0.05}, # Decrease: unnecessary
                1: {0: 0.05, 1: 0.3, 2: 0.6, 3: 0.05}, # Keep: stays good
                2: {0: 0.03, 1: 0.15, 2: 0.8, 3: 0.02} # Increase: remains excellent
            },
            # From S4 (Default) - terminal state
            3: {
                0: {3: 1.0},
                1: {3: 1.0},
                2: {3: 1.0}
            }
        }
        return transitions
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Start from S1, S2, or S3 (not S4)
        # Distribution: 20% S1, 50% S2, 30% S3
        self.current_state = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
        self.steps = 0
        
        return self.current_state, {}
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, truncated, info)
        """
        if self.current_state == 3:  # Already in terminal state
            return self.current_state, -20, True, False, {}
        
        # Get reward
        reward = self.reward_matrix[self.current_state][action]
        
        # Determine next state based on transition probabilities
        transition_dist = self.transition_probs[self.current_state][action]
        next_states = list(transition_dist.keys())
        probs = list(transition_dist.values())
        next_state = np.random.choice(next_states, p=probs)
        
        # Update state
        self.current_state = next_state
        self.steps += 1
        
        # Check if episode is done
        done = (next_state == 3)  # Terminal state (default)
        truncated = (self.steps >= self.max_steps)  # Max episode length
        
        info = {
            'state_name': self.state_names[next_state],
            'action_name': self.action_names[action],
            'steps': self.steps
        }
        
        return next_state, reward, done, truncated, info
    
    def render(self):
        """Print current state"""
        print(f"Current State: {self.state_names[self.current_state]}")
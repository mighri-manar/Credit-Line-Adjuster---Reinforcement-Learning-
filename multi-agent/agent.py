import numpy as np
import pickle

class QLearningAgent:
    """
    Q-Learning Agent for Credit Line Adjustment
    """
    
    def __init__(self, n_states=4, n_actions=3, 
                 learning_rate=0.2, discount_factor=0.95,
                 epsilon=0.2, epsilon_min=0.05, epsilon_decay=0.995):
        """
        Initialize Q-Learning Agent
        
        Args:
            n_states: Number of states (4)
            n_actions: Number of actions (3)
            learning_rate (alpha): Learning rate
            discount_factor (gamma): Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # For tracking learning
        self.episode_rewards = []
        self.episode_lengths = []
        
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: If False, always exploit (no exploration)
        
        Returns:
            action: Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-Learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state, action]
        
        if done:
            # Terminal state: no future reward
            target_q = reward
        else:
            # Non-terminal: use max Q-value of next state
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Q-Learning update
        td_error = target_q - current_q
        self.q_table[state, action] = current_q + self.alpha * td_error
        
        return td_error
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self):
        """Extract deterministic policy from Q-table"""
        return np.argmax(self.q_table, axis=1)
    
    def save(self, filepath='q_learning_agent.pkl'):
        """Save agent to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths
            }, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath='q_learning_agent.pkl'):
        """Load agent from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.episode_rewards = data['episode_rewards']
            self.episode_lengths = data['episode_lengths']
        print(f"Agent loaded from {filepath}")
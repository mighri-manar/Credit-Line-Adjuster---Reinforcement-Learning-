import numpy as np
import pickle
from typing import Dict, List, Tuple
from agent import QLearningAgent

class MultiQLearningAgent:
    """
    Multi-Agent Q-Learning System for Credit Line Adjustment
    Uses specialized agents for different risk states
    """
    
    def __init__(self, learning_rate=0.2, discount_factor=0.95,
                 epsilon=0.2, epsilon_min=0.05, epsilon_decay=0.995):
        """
        Initialize Multi-Agent Q-Learning System with specialized agents:
        - Agent 0: High Risk Specialist (S1)
        - Agent 1: Moderate Risk Specialist (S2) 
        - Agent 2: Low Risk Specialist (S3)
        - Agent 3: Default Handler (S4)
        """
        self.n_agents = 4
        self.n_actions = 3
        
        # Create specialized agents for each state
        self.agents = {}
        self.agent_names = {
            0: "High Risk Specialist",
            1: "Moderate Risk Specialist", 
            2: "Low Risk Specialist",
            3: "Default Handler"
        }
        
        for state in range(self.n_agents):
            self.agents[state] = QLearningAgent(
                n_states=4,  # All agents aware of all states
                n_actions=3,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay
            )
            
        # Coordination mechanism
        self.coordination_weights = np.ones(self.n_agents) / self.n_agents
        self.agent_performance = np.zeros(self.n_agents)
        self.agent_episode_counts = np.zeros(self.n_agents)
        
        # For tracking learning
        self.episode_rewards = []
        self.agent_contributions = {i: [] for i in range(self.n_agents)}
        
    def choose_action(self, state, training=True):
        """
        Choose action using multi-agent coordination
        Primary agent handles the decision, others provide input
        """
        primary_agent = self.agents[state]
        
        if training:
            # Primary agent makes decision with coordination from others
            primary_action = primary_agent.choose_action(state, training=True)
            
            # Collect suggestions from other agents (consensus mechanism)
            other_suggestions = []
            for agent_id, agent in self.agents.items():
                if agent_id != state:  # Not the primary agent
                    suggestion = agent.choose_action(state, training=False)
                    other_suggestions.append(suggestion)
            
            # Weighted coordination (primary agent has higher weight)
            if other_suggestions:
                consensus_action = max(set(other_suggestions), key=other_suggestions.count)
                
                # 80% primary agent, 20% consensus from others
                if np.random.random() < 0.8:
                    return primary_action
                else:
                    return consensus_action
            else:
                return primary_action
        else:
            # During testing, use the specialized agent for the state
            return primary_agent.choose_action(state, training=False)
    
    def update(self, state, action, reward, next_state, done):
        """
        Update all agents with different learning strategies
        """
        td_errors = []
        
        # Primary agent (responsible for this state) gets full update
        primary_agent = self.agents[state]
        primary_td_error = primary_agent.update(state, action, reward, next_state, done)
        td_errors.append(primary_td_error)
        
        # Update agent performance tracking
        self.agent_performance[state] += reward
        self.agent_episode_counts[state] += 1
        
        # Other agents learn from observation (reduced learning rate)
        for agent_id, agent in self.agents.items():
            if agent_id != state:
                # Reduced learning rate for observing agents
                original_alpha = agent.alpha
                agent.alpha *= 0.3  # 30% of original learning rate
                
                observational_td_error = agent.update(state, action, reward, next_state, done)
                td_errors.append(observational_td_error)
                
                # Restore original learning rate
                agent.alpha = original_alpha
        
        return td_errors
    
    def decay_epsilon(self):
        """Decay exploration rate for all agents"""
        for agent in self.agents.values():
            agent.decay_epsilon()
    
    def get_policy(self):
        """Extract policies from all specialized agents"""
        policies = {}
        for agent_id, agent in self.agents.items():
            policies[self.agent_names[agent_id]] = agent.get_policy()
        return policies
    
    def get_q_tables(self):
        """Get Q-tables from all agents"""
        q_tables = {}
        for agent_id, agent in self.agents.items():
            q_tables[self.agent_names[agent_id]] = agent.q_table.copy()
        return q_tables
    
    def get_agent_performance(self):
        """Get performance statistics for each agent"""
        performance = {}
        for agent_id in range(self.n_agents):
            if self.agent_episode_counts[agent_id] > 0:
                avg_performance = self.agent_performance[agent_id] / self.agent_episode_counts[agent_id]
            else:
                avg_performance = 0.0
            
            performance[self.agent_names[agent_id]] = {
                'total_reward': self.agent_performance[agent_id],
                'episodes': int(self.agent_episode_counts[agent_id]),
                'avg_reward': avg_performance
            }
        return performance
    
    def save(self, filepath='multi_agent_system.pkl'):
        """Save multi-agent system to file"""
        save_data = {
            'agents': {},
            'coordination_weights': self.coordination_weights,
            'agent_performance': self.agent_performance,
            'agent_episode_counts': self.agent_episode_counts,
            'episode_rewards': self.episode_rewards,
            'agent_contributions': self.agent_contributions
        }
        
        # Save each agent's state
        for agent_id, agent in self.agents.items():
            save_data['agents'][agent_id] = {
                'q_table': agent.q_table,
                'epsilon': agent.epsilon,
                'episode_rewards': agent.episode_rewards,
                'episode_lengths': agent.episode_lengths
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Multi-agent system saved to {filepath}")
    
    def load(self, filepath='multi_agent_system.pkl'):
        """Load multi-agent system from file"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.coordination_weights = save_data['coordination_weights']
        self.agent_performance = save_data['agent_performance']
        self.agent_episode_counts = save_data['agent_episode_counts']
        self.episode_rewards = save_data['episode_rewards']
        self.agent_contributions = save_data['agent_contributions']
        
        # Load each agent's state
        for agent_id, agent_data in save_data['agents'].items():
            self.agents[agent_id].q_table = agent_data['q_table']
            self.agents[agent_id].epsilon = agent_data['epsilon']
            self.agents[agent_id].episode_rewards = agent_data['episode_rewards']
            self.agents[agent_id].episode_lengths = agent_data['episode_lengths']
        
        print(f"Multi-agent system loaded from {filepath}")
    
    def print_agent_statistics(self):
        """Print detailed statistics for each specialized agent"""
        print("\n" + "="*80)
        print("MULTI-AGENT SYSTEM STATISTICS")
        print("="*80)
        
        performance = self.get_agent_performance()
        q_tables = self.get_q_tables()
        
        for agent_id in range(self.n_agents):
            agent_name = self.agent_names[agent_id]
            print(f"\n{agent_name} (Agent {agent_id}):")
            print(f"  Episodes Handled: {performance[agent_name]['episodes']}")
            print(f"  Total Reward: {performance[agent_name]['total_reward']:.2f}")
            print(f"  Average Reward: {performance[agent_name]['avg_reward']:.2f}")
            print(f"  Current Epsilon: {self.agents[agent_id].epsilon:.4f}")
            
            # Show Q-table for this agent
            print(f"  Q-Table:")
            q_table = q_tables[agent_name]
            print("         Decrease    Keep    Increase")
            state_names = ["S1", "S2", "S3", "S4"]
            for s in range(4):
                print(f"    {state_names[s]}: {q_table[s, 0]:8.2f} {q_table[s, 1]:8.2f} {q_table[s, 2]:8.2f}")
        
        # Overall coordination weights
        print(f"\nCoordination Weights:")
        for agent_id in range(self.n_agents):
            print(f"  {self.agent_names[agent_id]}: {self.coordination_weights[agent_id]:.3f}")


class AdaptiveMultiAgent(MultiQLearningAgent):
    """
    Advanced Multi-Agent system with adaptive coordination
    Agents' influence changes based on their performance
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adaptation_rate = 0.01
        self.min_weight = 0.1
        
    def update_coordination_weights(self):
        """Update coordination weights based on agent performance"""
        if np.sum(self.agent_episode_counts) > 100:  # After some experience
            # Calculate performance-based weights
            avg_performances = []
            for agent_id in range(self.n_agents):
                if self.agent_episode_counts[agent_id] > 0:
                    avg_perf = self.agent_performance[agent_id] / self.agent_episode_counts[agent_id]
                else:
                    avg_perf = 0.0
                avg_performances.append(avg_perf)
            
            # Normalize to weights (better performing agents get higher weights)
            if max(avg_performances) > min(avg_performances):
                normalized_weights = np.array(avg_performances)
                normalized_weights = (normalized_weights - min(normalized_weights))
                normalized_weights = normalized_weights / max(normalized_weights) if max(normalized_weights) > 0 else normalized_weights
                
                # Ensure minimum weight and normalize
                normalized_weights = np.maximum(normalized_weights, self.min_weight)
                normalized_weights = normalized_weights / np.sum(normalized_weights)
                
                # Smoothly update weights
                self.coordination_weights = (1 - self.adaptation_rate) * self.coordination_weights + \
                                          self.adaptation_rate * normalized_weights
    
    def update(self, state, action, reward, next_state, done):
        """Enhanced update with adaptive coordination"""
        td_errors = super().update(state, action, reward, next_state, done)
        
        # Update coordination weights periodically
        if np.sum(self.agent_episode_counts) % 50 == 0:
            self.update_coordination_weights()
        
        return td_errors
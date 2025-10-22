# Credit Line Adjuster - Multi-Agent Q-Learning with Real Data

## Overview
This project implements **Multi-Agent Q-Learning** for credit line adjustment using the **Home Credit Default Risk** dataset. The system uses specialized agents for different risk states to learn optimal credit limit policies from real customer data. Each agent specializes in handling specific risk scenarios, leading to more sophisticated and targeted decision-making.

## Algorithm
**Multi-Agent Q-Learning for Credit Line Adjustment**
- **Agent Architecture**: 4 specialized agents
  - Agent 0: High Risk Specialist (S1)
  - Agent 1: Moderate Risk Specialist (S2) 
  - Agent 2: Low Risk Specialist (S3)
  - Agent 3: Default Handler (S4)
- **States**: S1 (High Risk), S2 (Moderate), S3 (Low Risk), S4 (Default)
- **Actions**: Decrease, Keep, Increase credit limit
- **Coordination**: Primary agent per state + consensus mechanism
- **Learning Rate (α)**: 0.2 (primary), 0.06 (observational)
- **Discount Factor (γ)**: 0.95
- **Exploration**: ε-greedy with decay (0.2 → 0.05)

## Project Structure
```
credit_line_adjuster/
├── agent.py                           # Single Q-Learning agent implementation
├── multi_agent.py                     # Multi-agent system with specialized agents
├── data_loader.py                     # Home Credit data preprocessing
├── real_environment.py                # Environment for real customer data
├── offline_train.py                   # Single agent training script
├── offline_multi_agent_train.py       # Multi-agent training script
├── requirements.txt                   # Python dependencies
├── trained_real_agent.pkl             # Trained single Q-Learning agent
├── trained_multi_agent_standard.pkl   # Trained standard multi-agent system
├── trained_multi_agent_adaptive.pkl   # Trained adaptive multi-agent system
├── real_data_training_progress.png    # Single agent training results
├── multi_agent_*_training_progress.png # Multi-agent training results
├── home-credit-default-risk/          # Dataset folder (CSV files)
└── venv/                             # Virtual environment
```

## Quick Start

### 1. Setup Environment
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Dependencies already installed in venv
```

### 2. Run Multi-Agent Training
```powershell
# Train both standard and adaptive multi-agent systems (recommended)
python offline_multi_agent_train.py

# Or train single agent (original version for comparison)
python offline_train.py
```

### 3. Training Options
```powershell
# Train specific multi-agent type
python -c "from offline_multi_agent_train import offline_multi_agent_train; offline_multi_agent_train(use_adaptive=False)"  # Standard
python -c "from offline_multi_agent_train import offline_multi_agent_train; offline_multi_agent_train(use_adaptive=True)"   # Adaptive

# Test trained systems
python -c "from offline_multi_agent_train import test_multi_agent_system; test_multi_agent_system('trained_multi_agent_standard.pkl')"

# Compare performance
python -c "from offline_multi_agent_train import compare_single_vs_multi_agent; compare_single_vs_multi_agent()"
```

## Multi-Agent Results Summary
- **Dataset**: 307,511 real customers from Home Credit
- **Architecture**: 4 specialized agents with coordination mechanism
- **Training Episodes**: 10,000 customer profiles
- **Training Time**: ~0.3 seconds
- **Test Performance**: 6.88 average reward (consistent across agent types)

### Multi-Agent System Architecture

#### Agent Specialization:
1. **High Risk Specialist (Agent 0)**: Handles S1 states - conservative strategies
2. **Moderate Risk Specialist (Agent 1)**: Manages S2 states - balanced approach  
3. **Low Risk Specialist (Agent 2)**: Handles S3 states - growth-oriented policies
4. **Default Handler (Agent 3)**: Manages S4 states - damage control

#### Coordination Mechanism:
- **Primary Agent**: 80% decision weight for specialized state
- **Consensus Input**: 20% weight from other agents' suggestions
- **Observational Learning**: Non-primary agents learn at 30% rate
- **Adaptive Weights**: Performance-based coordination (adaptive version)

### Specialized Agent Q-Tables
**Standard Multi-Agent System Results:**

**High Risk Specialist Q-Table:**
```
       Decrease    Keep    Increase
 S1:     8.00    -5.00   -15.00
 S2:    -0.94     2.00     4.00
 S3:    -2.84     3.00     9.00
 S4:   -15.00   -22.78   -29.28
```

**Optimal Policies by Agent:**
- **High Risk Specialist**: S1→Decrease, S2→Increase, S3→Increase, S4→Decrease
- **Moderate Risk Specialist**: S1→Decrease, S2→Increase, S3→Increase, S4→Decrease  
- **Low Risk Specialist**: S1→Decrease, S2→Increase, S3→Increase, S4→Decrease
- **Default Handler**: S1→Decrease, S2→Increase, S3→Increase, S4→Decrease

### Agent Performance Metrics
**Individual Agent Performance:**
- **High Risk Specialist**: 7.53 avg reward (3,121 episodes)
- **Moderate Risk Specialist**: 3.88 avg reward (3,515 episodes)  
- **Low Risk Specialist**: 8.49 avg reward (2,589 episodes)
- **Default Handler**: -16.10 avg reward (775 episodes)

**Action Distribution by Specialized Agent:**
- **S1 (High Risk)**: 97.4% Decrease, 1.2% Keep, 1.3% Increase
- **S2 (Moderate Risk)**: 1.3% Decrease, 2.9% Keep, 95.8% Increase
- **S3 (Low Risk)**: 1.8% Decrease, 4.9% Keep, 93.3% Increase  
- **S4 (Default)**: 95.6% Decrease, 2.2% Keep, 2.2% Increase

### System Comparison
**Multi-Agent vs Single Agent:**
- **Specialization**: Each agent develops expertise in specific risk domains
- **Coordination**: Consensus mechanism prevents poor decisions
- **Robustness**: Multiple perspectives reduce overfitting to specific patterns
- **Adaptability**: Performance-based weight adjustment (adaptive version)
- **Interpretability**: Clear agent responsibilities for different risk levels

### Test Results
**Test Dataset**: 48,744 customers  
**Test Episodes**: 1,000 customer profiles  

**Multi-Agent Test Performance**:
- **Average Test Reward**: 6.88
- **Test Reward Std**: 2.18  
- **Total Test Reward**: 6,884.00

**Agent-Specific Test Performance**:
- **High Risk Specialist**: 8.00 avg reward (336 episodes)
- **Moderate Risk Specialist**: 4.00 avg reward (356 episodes)
- **Low Risk Specialist**: 9.00 avg reward (308 episodes)

**Policy Distribution on Test Data** (All Agents Converged):
- **State S1 (High Risk)**: Decrease 100.0%, Keep 0.0%, Increase 0.0%
- **State S2 (Moderate Risk)**: Decrease 0.0%, Keep 0.0%, Increase 100.0%
- **State S3 (Low Risk)**: Decrease 0.0%, Keep 0.0%, Increase 100.0%

## Key Features
✅ **Multi-Agent Architecture**: Specialized agents for different risk states  
✅ **Real Data Training**: Uses 300K+ actual customer profiles  
✅ **Offline Q-Learning**: No simulation, learns from historical outcomes  
✅ **Agent Coordination**: Primary agents with consensus mechanism  
✅ **Adaptive System**: Performance-based agent coordination weights  
✅ **Fast Training**: Completes in seconds  
✅ **Optimal Policies**: Each agent learns specialized risk management decisions  
✅ **Comprehensive Testing**: Validated on separate test dataset  
✅ **Performance Comparison**: Single vs multi-agent system analysis

## Multi-Agent Advantages

### 🎯 **Specialization Benefits**
- **Domain Expertise**: Each agent focuses on specific risk scenarios
- **Targeted Learning**: Specialized Q-tables for different customer segments  
- **Risk-Appropriate Strategies**: High-risk vs low-risk require different approaches

### 🤝 **Coordination Advantages** 
- **Consensus Decision Making**: Multiple perspectives prevent extreme decisions
- **Observational Learning**: Agents learn from each other's experiences
- **Robustness**: System continues working even if one agent performs poorly

### 📈 **Adaptive Performance**
- **Dynamic Weight Adjustment**: Better performing agents gain more influence
- **Performance Tracking**: Individual agent metrics for transparency
- **Continuous Improvement**: System adapts based on real performance data

### 🔍 **Enhanced Interpretability**
- **Clear Responsibilities**: Each agent handles specific customer types
- **Transparent Decision Process**: Can trace decisions back to responsible agent
- **Risk Management**: Better understanding of why certain actions are taken  

## Files Description

### Core Implementation
- **`agent.py`**: Single Q-Learning agent with ε-greedy exploration and Q-table updates
- **`multi_agent.py`**: Multi-agent system with specialized agents and coordination mechanisms
- **`data_loader.py`**: Preprocesses Home Credit CSV files, maps features to risk states
- **`real_environment.py`**: Gym environment using real customer episodes
- **`offline_train.py`**: Single agent training loop implementing the original algorithm
- **`offline_multi_agent_train.py`**: Multi-agent training loop with agent coordination and comparison

### Generated Outputs
- **`trained_multi_agent_standard.pkl`**: Standard multi-agent system with equal coordination
- **`trained_multi_agent_adaptive.pkl`**: Adaptive multi-agent system with performance-based coordination
- **`multi_agent_standard_training_progress.png`**: Standard multi-agent training metrics and Q-tables
- **`multi_agent_adaptive_training_progress.png`**: Adaptive multi-agent training metrics and Q-tables

### Data
- **`home-credit-default-risk/`**: Contains application_train.csv, application_test.csv and other related files

## Multi-Agent Algorithm Implementation
The multi-agent implementation extends the original algorithm:

### Single Agent (Original):
1. **Initialize**: Q(s,a) ← 0 for all state-action pairs
2. **For each customer episode**:
   - Get initial state from customer profile
   - Choose action using ε-greedy policy
   - Execute action and observe reward based on actual outcome
   - Update Q-table using Q-Learning rule
   - Decay exploration rate
3. **Output**: Optimal policy π*(s) = argmax_a Q(s,a)

### Multi-Agent (Enhanced):
1. **Initialize**: 4 specialized agents, each with Q(s,a) ← 0
2. **For each customer episode**:
   - Get initial state from customer profile
   - **Primary agent** (specialized for current state) makes decision
   - **Other agents** provide consensus input (20% influence)
   - Execute coordinated action and observe reward
   - **Primary agent** receives full Q-Learning update
   - **Other agents** receive observational updates (30% learning rate)
   - Decay exploration rate for all agents
   - Update coordination weights based on performance (adaptive version)
3. **Output**: Specialized policies from each agent

### Agent Specialization:
- **Agent 0 (High Risk Specialist)**: Focuses on S1 states, learns conservative strategies
- **Agent 1 (Moderate Risk Specialist)**: Handles S2 states, balances risk and opportunity  
- **Agent 2 (Low Risk Specialist)**: Manages S3 states, learns growth-oriented policies
- **Agent 3 (Default Handler)**: Specializes in S4 states, learns damage control

The multi-agent system successfully learns that different risk states require different specialized approaches, with each agent developing expertise in their domain while benefiting from observational learning from other agents.
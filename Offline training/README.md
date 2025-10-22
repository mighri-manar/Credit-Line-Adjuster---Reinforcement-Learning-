# Credit Line Adjuster - Offline Q-Learning with Real Data

## Overview
This project implements **offline Q-Learning** for credit line adjustment using the **Home Credit Default Risk** dataset. The system learns optimal credit limit policies from real customer data instead of simulation.

## Algorithm
**Q-Learning for Credit Line Adjustment**
- **States**: S1 (High Risk), S2 (Moderate), S3 (Low Risk), S4 (Default)
- **Actions**: Decrease, Keep, Increase credit limit
- **Learning Rate (α)**: 0.2
- **Discount Factor (γ)**: 0.95
- **Exploration**: ε-greedy with decay (0.2 → 0.05)

## Project Structure
```
credit_line_adjuster/
├── agent.py                           # Q-Learning agent implementation
├── data_loader.py                     # Home Credit data preprocessing
├── real_environment.py                # Environment for real customer data
├── offline_train.py                   # Main offline training script
├── requirements.txt                   # Python dependencies
├── trained_real_agent.pkl             # Trained Q-Learning agent
├── real_data_training_progress.png    # Training results visualization
├── home-credit-default-risk/          # Dataset folder (CSV files)
└── venv/                             # Virtual environment
```

## Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Dependencies already installed in venv
```

### 2. Run Offline Training
```bash
python offline_train.py
```

## Results Summary
- **Dataset**: 307,511 real customers from Home Credit
- **Training Episodes**: 10,000 customer profiles
- **Training Time**: 0.15 seconds
- **Final Performance**: 4.81 average reward
- **Test Performance**: 6.88 average reward

### Learned Q-Table
**States**: S1=High Risk, S2=Moderate, S3=Low Risk, S4=Default  
**Actions**: 0=Decrease, 1=Keep, 2=Increase

```
       Decrease    Keep    Increase
 S1:   172.79   157.58   147.80
 S2:   164.10   165.31   173.60
 S3:   166.55   172.81   180.00
 S4:   -15.00   -34.01   -42.53
```

### Optimal Policy π*(s) = argmax_a Q(s,a)
- **S1 (High Risk - High Utilization/Poor History)**: Decrease (Q=172.79)
- **S2 (Moderate Risk - Moderate/Stable)**: Increase (Q=173.60)
- **S3 (Low Risk - Low Utilization/Strong History)**: Increase (Q=180.00)
- **S4 (Default)**: Decrease (Q=-15.00)

### Action Frequency by State
**S1 (High Risk - High Utilization/Poor History)**:
- Decrease: 96.3% (avg reward: 8.00)
- Keep: 2.0% (avg reward: -5.00)
- Increase: 1.7% (avg reward: -15.00)

**S2 (Moderate Risk - Moderate/Stable)**:
- Decrease: 1.9% (avg reward: -1.00)
- Keep: 14.1% (avg reward: 2.00)
- Increase: 84.0% (avg reward: 4.00)

**S3 (Low Risk - Low Utilization/Strong History)**:
- Decrease: 1.4% (avg reward: -3.00)
- Keep: 31.5% (avg reward: 3.00)
- Increase: 67.1% (avg reward: 9.00)

**S4 (Default)**:
- Decrease: 96.3% (avg reward: -15.00)
- Keep: 2.1% (avg reward: -35.00)
- Increase: 1.7% (avg reward: -45.00)

### Performance Metrics
- **Total Episodes**: 10,000
- **Total Reward**: 41,308.00
- **Average Reward per Episode**: 4.13
- **Final 1000 Episodes Avg Reward**: 4.81
- **Standard Deviation**: 6.69
- **Final Epsilon**: 0.0500

### Test Results
**Test Dataset**: 48,744 customers  
**Test Episodes**: 1,000 customer profiles  

**State Distribution in Test Data**:
- S1 (High Risk): 336 customers (33.6%)
- S2 (Moderate Risk): 356 customers (35.6%)
- S3 (Low Risk): 308 customers (30.8%)

**Test Performance**:
- **Average Test Reward**: 6.88
- **Test Reward Std**: 2.18
- **Total Test Reward**: 6,884.00

**Policy Distribution on Test Data**:
- **State S1 (High Risk)**: Decrease 100.0%, Keep 0.0%, Increase 0.0%
- **State S2 (Moderate Risk)**: Decrease 0.0%, Keep 0.0%, Increase 100.0%
- **State S3 (Low Risk)**: Decrease 0.0%, Keep 0.0%, Increase 100.0%

## Key Features
✅ **Real Data Training**: Uses 300K+ actual customer profiles  
✅ **Offline Q-Learning**: No simulation, learns from historical outcomes  
✅ **Fast Training**: Completes in seconds  
✅ **Optimal Policy**: Learns sound risk management decisions  
✅ **Comprehensive Testing**: Validated on separate test dataset  

## Files Description

### Core Implementation
- **`agent.py`**: Q-Learning agent with ε-greedy exploration and Q-table updates
- **`data_loader.py`**: Preprocesses Home Credit CSV files, maps features to risk states
- **`real_environment.py`**: Gym environment using real customer episodes
- **`offline_train.py`**: Main training loop implementing the specified algorithm

### Generated Outputs
- **`trained_real_agent.pkl`**: Serialized trained agent with learned Q-table
- **`real_data_training_progress.png`**: Training metrics and policy visualization

### Data
- **`home-credit-default-risk/`**: Contains application_train.csv, application_test.csv and other related files

## Algorithm Implementation
The implementation follows the exact specification:

1. **Initialize**: Q(s,a) ← 0 for all state-action pairs
2. **For each customer episode**:
   - Get initial state from customer profile
   - Choose action using ε-greedy policy
   - Execute action and observe reward based on actual outcome
   - Update Q-table using Q-Learning rule
   - Decay exploration rate
3. **Output**: Optimal policy π*(s) = argmax_a Q(s,a)

The system successfully learns that:
- High-risk customers should have credit **decreased**
- Moderate-risk customers should have credit **increased** 
- Low-risk customers should have credit **increased**
- Defaulted customers should have credit **decreased**
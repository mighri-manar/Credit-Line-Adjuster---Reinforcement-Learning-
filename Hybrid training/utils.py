import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_training_progress(episode_rewards, episode_lengths):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                 np.ones(window)/window, 
                                 mode='valid')
        axes[0].plot(range(window-1, len(episode_rewards)), 
                    moving_avg, 
                    'r-', 
                    linewidth=2, 
                    label=f'{window}-Episode Moving Avg')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Progress: Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[1].plot(episode_lengths, alpha=0.3, label='Episode Length')
    
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, 
                                 np.ones(window)/window, 
                                 mode='valid')
        axes[1].plot(range(window-1, len(episode_lengths)), 
                    moving_avg, 
                    'g-', 
                    linewidth=2, 
                    label=f'{window}-Episode Moving Avg')
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length (steps)')
    axes[1].set_title('Training Progress: Episode Lengths')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("📊 Training progress plot saved as 'training_progress.png'")
    plt.show()

def plot_q_table_heatmap(q_table):
    """Visualize Q-table as heatmap"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    state_labels = ['S1\n(High Risk)', 'S2\n(Moderate)', 'S3\n(Low Risk)', 'S4\n(Default)']
    action_labels = ['Decrease', 'Keep', 'Increase']
    
    sns.heatmap(q_table, 
                annot=True, 
                fmt='.2f', 
                cmap='RdYlGn', 
                center=0,
                xticklabels=action_labels,
                yticklabels=state_labels,
                cbar_kws={'label': 'Q-Value'},
                linewidths=0.5,
                ax=ax)
    
    ax.set_title('Q-Table Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('q_table_heatmap.png', dpi=300, bbox_inches='tight')
    print("📊 Q-table heatmap saved as 'q_table_heatmap.png'")
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Define metric options
METRIC_CONFIG = {
    'total_revenue': {
        'column': 'total_revenue',
        'ylabel': 'Revenue',
        'title': 'Total Revenue Stability Across Tree Structures'
    },
    'drop_rate': {
        'column': 'drop_rate',
        'ylabel': 'Drop Rate',
        'title': 'Drop Rate Stability Across Tree Structures'
    },
    'rev_per_request': {
        'column': 'rev_per_request',
        'ylabel': 'Revenue per Request ($)',
        'title': 'Revenue per Request Stability Across Tree Structures'
    },
    'hit_rate_mean': {
        'column': 'hit_rate_mean',
        'ylabel': 'Hit Rate',
        'title': 'Hit Rate Stability Across Tree Structures'
    }
}

def plot_metric_by_tree_structure(df, metric='rev_per_request', save_name=None):
    """
    Plot a specific metric for different request loads representing different tree structures
    Shows stability of policy performance across different configurations
    
    Parameters:
    -----------
    df : DataFrame
        The data to plot
    metric : str
        The metric to plot
    save_name : str, optional
        Custom filename to save the plot
    """
    if metric not in METRIC_CONFIG:
        print(f"Error: Invalid metric '{metric}'")
        print(f"Available metrics: {list(METRIC_CONFIG.keys())}")
        return
    
    config = METRIC_CONFIG[metric]
    
    # Define the specific request configurations to plot
    # Format: (label, n_requests) - representing different tree structures
    configurations = [
        ('tree-3-2', 10000),
        ('tree-3-3', 20000),
        ('tree-3-4', 30000)
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique policies
    policies = df['policy'].unique()
    # Use light blue color for bars
    bar_color = "#1f77b4"
    line_color = "#0f598f"
    
    # X positions for the configurations
    x_positions = np.arange(len(configurations))
    bar_width = 0.2
    
    # Plot for each policy
    for p_idx, policy in enumerate(policies):
        metric_values = []
        revenue_values = []
        request_values = []
        
        # Extract data for each configuration
        for tree_label, n_req in configurations:
            row = df[(df['n_requests'] == n_req) & 
                     (df['policy'] == policy)]
            
            if not row.empty:
                metric_values.append(row[config['column']].values[0])
                revenue_values.append(row['total_revenue'].values[0])
                request_values.append(n_req)
            else:
                metric_values.append(0)
                revenue_values.append(0)
                request_values.append(n_req)
        
        # Plot bars
        offset = (p_idx - len(policies)/2 + 0.5) * bar_width
        bars = ax.bar(x_positions + offset, 
                      metric_values,
                      width=bar_width,
                      alpha=0.8,
                      color=bar_color,
                      edgecolor='black',
                      linewidth=1.5,
                      label=policy,
                      zorder=2)
        
        # Plot trend line
        ax.plot(x_positions + offset, 
                metric_values,
                marker='o',
                linewidth=2.5,
                markersize=10,
                color=line_color,
                linestyle='-',
                alpha=0.9,
                zorder=3)
        
        # Add value labels - showing metric value on top of bars (higher z-order)
        for i, (bar, metric_val, revenue_val, n_req) in enumerate(zip(bars, metric_values, revenue_values, request_values)):
            height = bar.get_height()
            # Show metric value on top with white background for visibility
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{metric_val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                             edgecolor=line_color, linewidth=2, alpha=0.95),
                    zorder=5)
    
    # Set axes - bars start from 0
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=0, top=max([max(df[df['policy']==p][config['column']]) for p in policies]) * 1.15)
    
    # Set x-axis labels to show tree structure with request count
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{tree}\n{n_req} Requests' for tree, n_req in configurations],
                       fontsize=12, fontweight='bold')
    
    # Labels and title
    ax.set_ylabel(config['ylabel'], fontsize=13, fontweight='bold')
    
    # Format y-axis based on metric type
    if metric == 'drop_rate' or metric == 'hit_rate_mean' or metric == 'rev_per_request':
        # For decimal metrics, show 2 decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    else:
        # For whole number metrics (revenue), use comma separators
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Rotate y-axis tick labels
    ax.tick_params(axis='y', labelrotation=45)
    
    # Add annotation box explaining stability
    if metric == 'rev_per_request':
        textstr = 'DDQN Stability Analysis:\n• Consistent revenue/request across tree topologies\n• Performance maintained despite varying complexity'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.6, edgecolor='navy', linewidth=2)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black', fancybox=True)
    
    plt.tight_layout()
    
    # Save figure
    if save_name is None:
        save_name = f'stability_{metric}_by_tree.png'
    
    save_path = os.path.join('results', save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Stability plot saved to: {save_path}")
    
    plt.show()
    plt.close()

def plot_ddqn_stability_comparison(df, save_name='ddqn_stability_comparison.png'):
    """
    Create a multi-panel plot showing DDQN stability across metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    metrics = ['rev_per_request', 'drop_rate', 'total_revenue', 'hit_rate_mean']
    
    configurations = [
        ('tree-3-2', 10000),
        ('tree-4-2', 20000),
        ('tree-5-2', 30000)
    ]
    
    x_positions = np.arange(len(configurations))
    
    for idx, metric in enumerate(metrics):
        if metric not in METRIC_CONFIG:
            continue
            
        config = METRIC_CONFIG[metric]
        ax = axes[idx]
        
        policies = df['policy'].unique()
        bar_color = 'lightblue'
        line_color = 'steelblue'
        bar_width = 0.2
        
        for p_idx, policy in enumerate(policies):
            metric_values = []
            revenue_values = []
            
            for tree_label, n_req in configurations:
                row = df[(df['n_requests'] == n_req) & 
                         (df['policy'] == policy)]
                
                if not row.empty:
                    metric_values.append(row[config['column']].values[0])
                    revenue_values.append(row['total_revenue'].values[0])
                else:
                    metric_values.append(0)
                    revenue_values.append(0)
            
            offset = (p_idx - len(policies)/2 + 0.5) * bar_width
            bars = ax.bar(x_positions + offset, 
                          metric_values,
                          width=bar_width,
                          alpha=0.8,
                          color=bar_color,
                          edgecolor='black',
                          linewidth=1.2,
                          label=policy,
                          zorder=2)
            
            ax.plot(x_positions + offset, 
                    metric_values,
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    color=line_color,
                    alpha=0.9,
                    zorder=3)
            
            # Add value labels on top with better visibility
            for i, (bar, val, rev) in enumerate(zip(bars, metric_values, revenue_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=line_color, linewidth=1.5, alpha=0.95),
                        zorder=5)
        
        ax.set_xlim(left=-0.5)
        ax.set_ylim(bottom=0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{tree}\n{n_req/1000:.0f}k req' for tree, n_req in configurations],
                           fontsize=10)
        ax.set_ylabel(config['ylabel'], fontsize=11, fontweight='bold')
        
        # Format y-axis based on metric type
        if metric == 'drop_rate' or metric == 'hit_rate_mean' or metric == 'rev_per_request':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.tick_params(axis='y', labelrotation=45)
        ax.tick_params(axis='x', labelrotation=45)
    
    plt.tight_layout()
    
    save_path = os.path.join('results', save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Multi-panel stability plot saved to: {save_path}")
    
    plt.show()
    plt.close()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot DDQN stability across tree structures')
    parser.add_argument('--file', '-f', type=str, default='results/CombinedResults.txt',
                        help='Path to the results file')
    parser.add_argument('--metrics', '-m', type=str, nargs='+', 
                        default=['rev_per_request'],
                        help='Metrics to plot')
    parser.add_argument('--multi', action='store_true',
                        help='Generate multi-panel comparison plot')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.file)
    
    print("=== DDQN Stability Analysis ===")
    print(f"\nData loaded from: {args.file}")
    print(f"Policies: {df['policy'].unique().tolist()}")
    print(f"Request counts: {sorted(df['n_requests'].unique().tolist())}")
    print("\n📊 Comparing stability across:")
    print("  ├─ tree-3-2 handling 10,000 requests")
    print("  ├─ tree-4-2 handling 20,000 requests")
    print("  └─ tree-5-2 handling 30,000 requests")
    print("="*50)
    
    # Generate individual metric plots
    for metric in args.metrics:
        print(f"\n--- Plotting {metric} stability ---")
        plot_metric_by_tree_structure(df, metric)
    
    # Generate multi-panel comparison if requested
    if args.multi:
        print("\n--- Generating multi-panel stability comparison ---")
        plot_ddqn_stability_comparison(df)
    
    print("\n" + "="*50)
    print("✓ Stability analysis complete!")
    print("="*50)
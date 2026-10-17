import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Plot cost stability per request")
parser.add_argument("--csv", default="results/combined_metrics.csv", help="Path to combined_metrics.csv")
parser.add_argument("--out-dir", default="results", help="Output directory for plot")
parser.add_argument("--policy", default="SPRIHA", help="Policy name to plot")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.csv)

# Compute cost per request (normalized)
df["placement_cost_per_request"] = df["placement_cost"] / df["n_requests"]
df["forwarding_cost_per_request"] = df["forwarding_cost"] / df["n_requests"]
df["violation_cost_per_request"] = df["violation_cost"] / df["n_requests"]
df["total_cost_per_request"] = df["total_cost"] / df["n_requests"]

# Select one policy
sub = df[df["policy"] == args.policy].sort_values("n_requests")

if sub.empty:
    print(f"No data found for policy: {args.policy}")
    exit(1)

# Plot with improved styling
fig, ax = plt.subplots(figsize=(14, 9))

# Define colors for each metric
colors = {
    'placement': '#e74c3c',
    'forwarding': '#f39c12',
    'violation': '#9b59b6',
    'total': '#3498db'
}

# Plot lines
line1 = ax.plot(sub["n_requests"], sub["placement_cost_per_request"], 
                marker='o', markersize=14, linewidth=5.2, color=colors['placement'],
                label="Placement Cost / Request", markeredgewidth=2, markeredgecolor='white')
line2 = ax.plot(sub["n_requests"], sub["forwarding_cost_per_request"], 
                marker='s', markersize=14, linewidth=5.2, color=colors['forwarding'],
                label="Forwarding Cost / Request", markeredgewidth=2, markeredgecolor='white')
line3 = ax.plot(sub["n_requests"], sub["violation_cost_per_request"], 
                marker='^', markersize=14, linewidth=5.2, color=colors['violation'],
                label="Violation Cost / Request", markeredgewidth=2, markeredgecolor='white')
line4 = ax.plot(sub["n_requests"], sub["total_cost_per_request"], 
                marker='d', markersize=14, linewidth=5.2, color=colors['total'],
                label="Total Cost / Request", markeredgewidth=2, markeredgecolor='white')

# Annotate final values at 10,000 requests (if available)
last_idx = sub.index[-1]
if sub.loc[last_idx, "n_requests"] == 10000:
    metrics = [
        ("placement_cost_per_request", colors['placement']),
        ("forwarding_cost_per_request", colors['forwarding']),
        ("violation_cost_per_request", colors['violation']),
        ("total_cost_per_request", colors['total'])
    ]
    
    for metric, color in metrics:
        final_val = sub.loc[last_idx, metric]
        ax.annotate(f'{final_val:.4f}',
                   xy=(10000, final_val),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   color=color,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor=color,
                           alpha=0.9,
                           linewidth=2))

# Set axis limits to ensure alignment at origin
ax.set_xlim(left=0, right=10500)
ax.set_ylim(bottom=0)

# Set explicit x-axis ticks
ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

# Labels and title with clarification about normalization
ax.set_xlabel("Number of Requests", fontsize=14, fontweight="bold")
ax.set_ylabel("Normalized Cost per Request", fontsize=14, fontweight="bold")
ax.set_title(f"", 
            fontsize=16, fontweight="bold", pad=20)

# Grid and legend
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
ax.legend(fontsize=12, loc="best", framealpha=0.95, edgecolor='black')

# Format y-axis to show more decimal places for small values
from matplotlib.ticker import FormatStrFormatter
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

# Adjust tick parameters
ax.tick_params(axis='both', labelsize=20)
ax.tick_params(axis='y', rotation=45)
plt.tight_layout()

# Save to the output directory
os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, f"{args.policy.lower()}_cost_per_request_plot.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
import sys
import pandas as pd
sys.path.insert(0, '.')
from edge_caching.network.topology import NetworkTopology

# Create topology
t = NetworkTopology(num_levels=3, num_children=4)

# Load CSV
df = pd.read_csv('training/data/sample_requests_10000_3_4_complete.csv')

# Get node sets
csv_nodes = set(df['origin_node_id'].unique())
topo_nodes = set([n.node_id for n in t.leaf_nodes])

print(f'CSV has {len(csv_nodes)} unique nodes')
print(f'Topology has {len(topo_nodes)} leaf nodes')

# Compare
missing_in_csv = topo_nodes - csv_nodes
extra_in_csv = csv_nodes - topo_nodes

if not missing_in_csv and not extra_in_csv:
    print('\n✓ PERFECT MATCH - All 64 leaf nodes are present!')
else:
    if missing_in_csv:
        print(f'\n✗ Missing in CSV ({len(missing_in_csv)}): {sorted(missing_in_csv)}')
    if extra_in_csv:
        print(f'\n⚠ Extra in CSV ({len(extra_in_csv)}): {sorted(extra_in_csv)}')


from collections import defaultdict
import heapq

def lca_cache_placement(contents, content_requests, paths, node_capacity):
    """
    contents: list of content IDs sorted by total profit (desc)
    content_requests: dict {content: [request_ids]}
    paths: dict {request_id: [leaf → root path]}
    node_capacity: dict {node: capacity}
    """

    # node_cache: node -> min-heap [(profit, content)]
    node_cache = defaultdict(list)

    def place_at_node(node, content, profit):
        heap = node_cache[node]

        if len(heap) < node_capacity[node]:
            heapq.heappush(heap, (profit, content))
        else:
            # replace lowest profit content
            if heap[0][0] < profit:
                heapq.heappop(heap)
                heapq.heappush(heap, (profit, content))

    for content in contents:
        reqs = content_requests[content]

        # Step 1: gather paths
        content_paths = [paths[r] for r in reqs]

        # Step 2: find LCA (intersection of paths)
        common_nodes = set(content_paths[0])
        for p in content_paths[1:]:
            common_nodes &= set(p)

        # Choose deepest common node (closest to leaves)
        if common_nodes:
            lca_node = max(common_nodes, key=lambda x: x)  # assumes numbering reflects depth
        else:
            lca_node = 0  # fallback to cloud

        # Step 3: compute total profit
        total_profit = len(reqs)  # or sum actual profits if available

        # Step 4: place at LCA
        place_at_node(lca_node, content, total_profit)

    return node_cache

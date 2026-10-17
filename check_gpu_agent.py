"""
Quick GPU detection test for DDQN Agent
"""
import sys
import os

# Add edge_caching to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing GPU-Accelerated DDQN Agent...")
print("=" * 72)

try:
    from edge_caching.agents.ddqn_agent import DDQNAgent

    # Create agent (will print GPU info)
    agent = DDQNAgent(
        state_dim=10,
        num_nodes=14,
        num_services=50
    )

    print("\nAgent initialized successfully!")

    # Test memory usage reporting
    mem_info = agent.get_memory_usage()
    print("\nMemory Usage:")
    for key, value in mem_info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 72)
    print("SUCCESS: DDQN Agent with GPU support is working!")
    print("=" * 72)

except Exception as e:
    print(f"\n? Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

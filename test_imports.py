"""
test_imports.py
---------------
Quick test to verify all imports work correctly after reorganization.
"""

print("Testing imports after reorganization...")
print("=" * 60)

try:
    print("\n1. Testing core imports...")
    from edge_caching.core import (
        EdgeNode, Service, Request,
        ACTION_INSERT, ACTION_DELETE, ACTION_KEEP,
        generate_service_catalogue
    )
    print("   ? Core imports successful")
    
    print("\n2. Testing network imports...")
    from edge_caching.network import NetworkTopology
    print("   ? Network imports successful")
    
    print("\n3. Testing agents imports...")
    from edge_caching.agents import DDQNAgent, QNetwork, ReplayBuffer
    print("   ? Agents imports successful")
    
    print("\n4. Testing control imports...")
    from edge_caching.control import EdgeOrchestrator
    print("   ? Control imports successful")
    
    print("\n5. Testing simulation imports...")
    from edge_caching.simulation import Simulation, RequestGenerator
    print("   ? Simulation imports successful")
    
    print("\n6. Testing utils imports...")
    from edge_caching.utils import save_all_results
    print("   ? Utils imports successful")
    
    print("\n7. Testing config imports...")
    from edge_caching import config
    print("   ? Config imports successful")
    
    print("\n8. Testing main package imports...")
    import edge_caching
    print(f"   ? Edge caching package version: {edge_caching.__version__}")
    
    print("\n" + "=" * 60)
    print("? All imports successful!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n? Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\n? Error: {e}")
    import traceback
    traceback.print_exc()

# Edge Caching Project - N-Ary Hierarchical Tree with DDQN Service Provisioning

"""
Edge Caching Project
====================

A hierarchical edge computing simulation with DDQN-based intelligent cache provisioning.

Project Structure
-----------------
- core/         : Domain models (Node, Service, Request)
- network/      : Network topology management
- agents/       : Reinforcement learning agents (DDQN)
- control/      : Orchestration and control logic
- simulation/   : Simulation engine and request generation
- utils/        : Visualization and helper utilities
- config.py     : Central configuration

Quick Start
-----------
>>> from edge_caching.utils.simulation import Simulation
>>> from edge_caching.utils import save_all_results
>>> 
>>> sim = Simulation()
>>> orchestrator = sim.run()
>>> save_all_results(orchestrator, sim.topology)
"""

__version__ = "1.0.0"

# Core components
from edge_caching.core import (
    ACTION_DELETE,
    ACTION_INSERT,
    ACTION_KEEP,
    EdgeNode,
    Request,
    Service,
    generate_service_catalogue,
)

# Network
from edge_caching.network import NetworkTopology

# Agents
from edge_caching.agents import DDQNAgent, QNetwork, ReplayBuffer

# Control
from edge_caching.control import EdgeOrchestrator

# Simulation
from edge_caching.utils.simulation import RequestGenerator, Simulation

# Utils
from edge_caching.utils import save_all_results

# Configuration
from edge_caching import config

__all__ = [
    # Core
    "EdgeNode",
    "Service",
    "Request",
    "ACTION_INSERT",
    "ACTION_DELETE",
    "ACTION_KEEP",
    "generate_service_catalogue",
    # Network
    "NetworkTopology",
    # Agents
    "DDQNAgent",
    "QNetwork",
    "ReplayBuffer",
    # Control
    "EdgeOrchestrator",
    # Simulation
    "Simulation",
    "RequestGenerator",
    # Utils
    "save_all_results",
    # Config
    "config",
]

"""
Core domain models for the edge caching simulation.
"""

from edge_caching.core.node import EdgeNode, ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.request import Request
from edge_caching.core.service import Service, generate_service_catalogue

__all__ = [
    "EdgeNode",
    "ACTION_INSERT",
    "ACTION_DELETE",
    "ACTION_KEEP",
    "Request",
    "Service",
    "generate_service_catalogue",
]


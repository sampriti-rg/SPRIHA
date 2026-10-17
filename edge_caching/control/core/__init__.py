"""
Core Domain Models
==================
Contains the fundamental domain entities:
- EdgeNode: Individual nodes in the edge network hierarchy
- Service: Cacheable services with attributes and metrics
- Request: User requests for services with deadlines
"""

from edge_caching.core.node import EdgeNode, ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.core.request import Request

__all__ = [
    "EdgeNode",
    "ACTION_INSERT",
    "ACTION_DELETE",
    "ACTION_KEEP",
    "Service",
    "generate_service_catalogue",
    "Request",
]

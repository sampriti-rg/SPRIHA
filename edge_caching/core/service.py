"""
service.py
----------
Defines the Service data class representing a cacheable service.

Counters (runtime, reset each provisioning interval)
-----------------------------------------------------
demand_count    : total requests received for this service across all nodes.
forward_count   : total times this service's request was forwarded (cache miss).
violation_count : total deadline violations (requests dropped).

Popularity update (EMA)
-----------------------
    popularity = alpha * observed_rate + (1 - alpha) * popularity
"""

import random
from dataclasses import dataclass, field
from edge_caching.config import CRITICALITY_WEIGHTS, POPULARITY_EMA_ALPHA


@dataclass
class Service:
    """
    A single service in the catalogue.

    Attributes
    ----------
    service_id      : unique identifier
    size            : actual service size in MB
    popularity      : current demand estimate [0.05, 1.0] - updated by EMA
    criticality     : tier label mapped to omega weight
    revenue         : base profit per successful local execution
    init_cpu_cycles : cold-start CPU cycles consumed on INSERT
    """
    service_id       : int
    size             : float
    popularity       : float
    criticality      : str
    revenue          : float
    init_cpu_cycles  : float = field(default=50_000)

    demand_count    : int   = field(default=0, compare=False, repr=False)
    forward_count   : int   = field(default=0, compare=False, repr=False)
    violation_count : int   = field(default=0, compare=False, repr=False)

    @property
    def omega(self) -> float:
        """Criticality weight omega(s) from CRITICALITY_WEIGHTS."""
        return CRITICALITY_WEIGHTS[self.criticality]

    def record_demand(self) -> None:
        self.demand_count += 1

    def record_forward(self) -> None:
        self.forward_count += 1

    def record_violation(self) -> None:
        self.violation_count += 1

    def update_popularity(self, total_requests_in_interval: int) -> None:
        """Update popularity using EMA of observed demand fraction."""
        observed_rate = self.demand_count / max(total_requests_in_interval, 1)
        self.popularity = (
            POPULARITY_EMA_ALPHA * observed_rate
            + (1 - POPULARITY_EMA_ALPHA) * self.popularity
        )
        self.popularity = max(0.05, min(1.0, self.popularity))

    def reset_interval_counters(self) -> None:
        """Reset all runtime counters. Called by orchestrator each interval."""
        self.demand_count    = 0
        self.forward_count   = 0
        self.violation_count = 0

    def drift_popularity(self, delta: float = 0.05) -> None:
        """Randomly perturb popularity to simulate exogenous demand shifts."""
        change = random.uniform(-delta, delta)
        self.popularity = max(0.05, min(1.0, self.popularity + change))

    @property
    def sla_violation_rate(self) -> float:
        """Fraction of demands that resulted in a deadline violation."""
        if self.demand_count == 0:
            return 0.0
        return self.violation_count / self.demand_count

    def __repr__(self) -> str:
        return (
            f"Service(id={self.service_id}, size={self.size:.0f}MB, "
            f"pop={self.popularity:.3f}, crit={self.criticality}, "
            f"rev={self.revenue:.2f}, omega={self.omega}, "
            f"init_cycles={self.init_cpu_cycles:.0f}, "
            f"demand={self.demand_count}, fwd={self.forward_count}, "
            f"violations={self.violation_count})"
        )


def generate_service_catalogue(num_services: int, seed: int = 42) -> list:
    """
    Generate a fixed catalogue of services with randomised initial attributes.

    Parameters
    ----------
    num_services : total number of services
    seed         : reproducibility seed

    Returns
    -------
    List of Service objects (runtime counters all start at 0)
    """
    size_mb_ranges = [(2, 150), (150, 1000), (1000, 4500)]
    init_ranges = [
        (10_000,     100_000),
        (100_000,  1_000_000),
        (1_000_000, 8_000_000),
    ]
    random.seed(seed)
    tiers     = list(CRITICALITY_WEIGHTS.keys())
    catalogue = []
    for i in range(num_services):
        tier    = random.randint(0, 2)
        size_mb = round(random.uniform(*size_mb_ranges[tier]), 1)
        svc     = Service(
            service_id      = i,
            size            = size_mb,
            popularity      = round(random.uniform(0.1, 1.0), 2),
            criticality     = tiers[i % len(tiers)],
            revenue         = round(random.uniform(1.0, 10.0), 2),
            init_cpu_cycles = round(random.uniform(*init_ranges[tier]), 0),
        )
        catalogue.append(svc)
    return catalogue

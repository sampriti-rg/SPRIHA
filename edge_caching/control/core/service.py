"""
service.py
----------
Defines the Service data class representing a cacheable service.

Counters (runtime, reset each provisioning interval)
-----------------------------------------------------
demand_count    : total requests received for this service across all nodes.
                  Each time ANY node receives a request for this service
                  (from a user), demand_count is incremented.
                  Used to update popularity via EMA at the end of each interval.

forward_count   : total times this service's request was forwarded (cache miss)
                  to a peer or parent node.  Drives the forwarding cost term
                  in the DDQN reward and the SLA-adjusted state vector.

violation_count : total deadline violations (requests dropped because
                  remaining_deadline <= 0 during routing).  Used as the
                  SLA violation signal in the reward penalty and state vector.

Popularity update (EMA)
-----------------------
At the end of every provisioning interval the orchestrator calls
update_popularity(observed_rate) to smooth observed demand into the
standing popularity value using Exponential Moving Average:

    popularity = alpha * observed_rate + (1 - alpha) * popularity

where alpha = POPULARITY_EMA_ALPHA (config.py).

This means popularity is no longer a fixed catalogue attribute - it
adapts to real request traffic and feeds directly into:
  - The DDQN state vector  (popularity slot)
  - The DDQN reward        (revenue term = popularity * revenue * omega * delta_exe)
  - The cache priority score (used by cache_initialiser for warm-start ranking)
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
    service_id    : unique identifier
    size          : storage units required (1 - 3)
    popularity    : current demand estimate [0.05, 1.0] - updated by EMA
    criticality   : tier label mapped to omega weight
    revenue       : base profit per successful local execution

    Runtime counters (reset each provisioning interval)
    ---------------------------------------------------
    demand_count    : requests received for this service (all nodes combined)
    forward_count   : cache-miss forwards for this service
    violation_count : deadline violations (drops) for this service
    """
    service_id       : int
    size             : float   # actual service size in MB  (real deployment footprint)
    popularity       : float   # EMA demand fraction     [0.05, 1.0]
    criticality      : str     # tier label -> omega weight
    revenue          : float   # profit per local execution  [cost/request]
    init_cpu_cycles  : float = field(default=50_000)
    # CPU cycles consumed ONCE when the service is cold-started after INSERT.
    # Covers: binary loading, memory allocation, runtime startup.
    # Drives T_init = init_cpu_cycles / node.cpu_capacity  [ms].
    # Calibrated range: 10_000 (static cache) to 8_000_000 (ML inference).
    # Source: Manner et al. (2018) ICSA-C; Alibaba Cluster Trace 2018.

    # Runtime counters - not part of the catalogue definition,
    # initialised to 0 and reset each provisioning interval
    demand_count    : int   = field(default=0, compare=False, repr=False)
    forward_count   : int   = field(default=0, compare=False, repr=False)
    violation_count : int   = field(default=0, compare=False, repr=False)

    # -----------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------

    @property
    def omega(self) -> float:
        """Criticality weight omega(s) from CRITICALITY_WEIGHTS."""
        return CRITICALITY_WEIGHTS[self.criticality]

    # -----------------------------------------------------------------
    # Popularity management
    # -----------------------------------------------------------------

    def record_demand(self) -> None:
        """
        Increment demand_count when a request for this service arrives at
        any origin node.  Called by EdgeNode.receive_request() at the first
        (origin) hop.
        """
        self.demand_count += 1

    def record_forward(self) -> None:
        """
        Increment forward_count when a request for this service is forwarded
        (cache miss) to a peer or parent.  Called by EdgeNode._handle_cache_miss().
        """
        self.forward_count += 1

    def record_violation(self) -> None:
        """
        Increment violation_count when a request for this service is dropped
        due to deadline expiry.  Called by EdgeNode when a drop occurs.
        """
        self.violation_count += 1

    def update_popularity(self, total_requests_in_interval: int) -> None:
        """
        Update popularity using Exponential Moving Average (EMA).

        observed_rate = demand_count / max(total_requests_in_interval, 1)

        The observed rate represents this service's share of total traffic
        in the last provisioning interval.  EMA smooths the update:

            popularity = alpha * observed_rate + (1 - alpha) * popularity

        Parameters
        ----------
        total_requests_in_interval : total requests routed in this interval
                                     (across all services and nodes)
        """
        observed_rate = self.demand_count / max(total_requests_in_interval, 1)
        self.popularity = (
            POPULARITY_EMA_ALPHA * observed_rate
            + (1 - POPULARITY_EMA_ALPHA) * self.popularity
        )
        # Clamp to valid range
        self.popularity = max(0.05, min(1.0, self.popularity))

    def reset_interval_counters(self) -> None:
        """
        Reset all runtime counters at the start of a new provisioning interval.
        Called by the orchestrator after update_popularity() has been applied.
        """
        self.demand_count    = 0
        self.forward_count   = 0
        self.violation_count = 0

    def drift_popularity(self, delta: float = 0.05) -> None:
        """
        Randomly perturb popularity to simulate exogenous demand shifts.
        Used between episodes in the simulation loop.
        """
        change = random.uniform(-delta, delta)
        self.popularity = max(0.05, min(1.0, self.popularity + change))

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------

    @property
    def sla_violation_rate(self) -> float:
        """
        Fraction of demands that resulted in a deadline violation this interval.
        Returns 0.0 when demand_count == 0 (no traffic yet).
        """
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


def generate_service_catalogue(num_services: int, seed: int = 42) -> list[Service]:
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
    # init_cpu_cycles ranges per size tier, calibrated to
    # Manner et al. (2018) cold-start measurements:
    #   size=1 (lightweight) : 10_000 –  100_000 cycles
    #   size=2 (moderate)    : 100_000 – 1_000_000 cycles
    #   size=3 (heavy)       : 1_000_000 – 8_000_000 cycles
    # size_mb ranges calibrated per service complexity tier:
    #   small  (IoT, VoIP, DNS)     :    2 -   150 MB
    #   medium (video, AI inference):  150 - 1000 MB
    #   large  (4K video, heavy ML) : 1000 - 4500 MB
    size_mb_ranges  = [(2, 150), (150, 1000), (1000, 4500)]
    init_ranges     = [
        (10_000,     100_000),   # small
        (100_000,  1_000_000),   # medium
        (1_000_000, 8_000_000),  # large
    ]
    random.seed(seed)
    tiers     = list(CRITICALITY_WEIGHTS.keys())
    catalogue = []
    for i in range(num_services):
        tier    = random.randint(0, 2)   # 0=small, 1=medium, 2=large
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

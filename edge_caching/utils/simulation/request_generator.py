"""
request_generator.py
--------------------
Generates synthetic service requests for leaf nodes using realistic
traffic arrival models that directly match Internet edge-caching literature.

Traffic Models
--------------

poisson_zipf  (default, recommended for DDQN training)
    Inter-arrival times
        t_k ~ Exponential(1 / POISSON_ARRIVAL_RATE)

        A Poisson arrival process is the standard model for aggregate
        user traffic at an edge node.  Each request arrives independently;
        the memoryless property means burst/quiet periods emerge naturally
        from the exponential gaps.

    Service selection (Zipf)
        Zipf's law is the empirically verified popularity model for web
        content, video-on-demand, and edge services: a small number of
        services receive the vast majority of requests.

        At the start of each generate() call, services are sorted by their
        current EMA-updated popularity (highest first) to assign Zipf ranks.
        The unnormalised weight for rank r is:

            w(r) = 1 / r^alpha          alpha = ZIPF_ALPHA from config

        The weights are normalised to a probability distribution and used for
        random.choices().  Because popularity is updated by EMA each
        provisioning interval, the Zipf ranking shifts over training,
        simulating realistic demand drift.

uniform  (legacy, used for comparison / ablation)
    Inter-arrival times
        t_k ~ Uniform(1, 5) ms

    Service selection
        Weighted by raw popularity values (same as before).

Both models produce identical Request objects so no downstream changes are
needed regardless of which model is active.

Parameters driven by config.py
-------------------------------
TRAFFIC_MODEL        : "poisson_zipf" | "uniform"
POISSON_ARRIVAL_RATE : lambda (mean requests per ms)
ZIPF_ALPHA           : Zipf exponent s (higher = more skewed distribution)
"""

import math
import random
import numpy as np
from typing import List, Dict

from edge_caching.core.request import Request
from edge_caching.core.service import Service
from edge_caching.core.node import EdgeNode
from edge_caching.config import (
    RANDOM_SEED, TRAFFIC_MODEL, POISSON_ARRIVAL_RATE, ZIPF_ALPHA,
)

# Criticality-aware deadline ranges [ms].
# Must match training/data/generate_requests.py so the agent trains on the
# same deadline distribution it is evaluated against.
_DEADLINE_RANGES = {
    "mission_critical": (5.0,   40.0),
    "high":             (50.0,  150.0),
    "standard":         (100.0, 300.0),
    "best_effort":      (200.0, 500.0),
}


def _zipf_weights(services: List[Service], alpha: float) -> List[float]:
    """
    Compute Zipf probability weights for a list of services.

    Services are ranked by their current popularity attribute (highest = rank 1).
    The Zipf weight for rank r is:

        w(r) = 1 / r^alpha

    This is then normalised to sum to 1.0.

    Parameters
    ----------
    services : list of Service objects, in any order
    alpha    : Zipf exponent (ZIPF_ALPHA from config)

    Returns
    -------
    weights : list of floats (same order as input services)
    """
    # Sort by popularity descending to assign Zipf ranks 1, 2, 3, ...
    ranked = sorted(
        enumerate(services),
        key=lambda x: x[1].popularity,
        reverse=True,
    )
    # rank[i] = 1-based rank of services[i]
    rank = [0] * len(services)
    for zipf_rank, (original_idx, _) in enumerate(ranked, start=1):
        rank[original_idx] = zipf_rank

    raw    = [1.0 / (rank[i] ** alpha) for i in range(len(services))]
    total  = sum(raw)
    return [w / total for w in raw]


class RequestGenerator:
    """
    Generates requests arriving at leaf nodes of the topology.

    Traffic model is selected at construction via the `model` parameter
    (defaults to TRAFFIC_MODEL from config.py).

    Parameters
    ----------
    catalogue    : {service_id -> Service}
    leaf_nodes   : list of leaf-level EdgeNodes
    seed         : reproducibility seed
    model        : "poisson_zipf" or "uniform"
    arrival_rate : Poisson lambda (mean requests/ms); only used by poisson_zipf
    zipf_alpha   : Zipf exponent; only used by poisson_zipf
    """

    def __init__(
        self,
        catalogue    : Dict[int, Service],
        leaf_nodes   : List[EdgeNode],
        seed         : int   = RANDOM_SEED,
        model        : str   = TRAFFIC_MODEL,
        arrival_rate : float = POISSON_ARRIVAL_RATE,
        zipf_alpha   : float = ZIPF_ALPHA,
    ):
        self.catalogue    = catalogue
        self.leaf_nodes   = leaf_nodes
        self.model        = model
        self.arrival_rate = arrival_rate
        self.zipf_alpha   = zipf_alpha

        random.seed(seed)
        np.random.seed(seed)

        self._sim_time : float = 0.0   # simulated clock in ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        num_requests      : int,
        deadline_range_ms : tuple = (50, 300),
    ) -> List[Request]:
        """
        Generate `num_requests` Request objects.

        The traffic model (set at construction) determines:
          - How inter-arrival times between requests are drawn
          - How the service for each request is selected

        In both models, the origin leaf node is chosen uniformly at
        random, and the deadline is drawn uniformly from deadline_range_ms.

        Parameters
        ----------
        num_requests      : number of requests to generate
        deadline_range_ms : (min_deadline, max_deadline) in milliseconds

        Returns
        -------
        List of Request objects in arrival-time order
        """
        if self.model == "poisson_zipf":
            return self._generate_poisson_zipf(num_requests, deadline_range_ms)
        else:
            return self._generate_uniform(num_requests, deadline_range_ms)

    # ------------------------------------------------------------------
    # Poisson arrivals + Zipf service selection
    # ------------------------------------------------------------------

    def _generate_poisson_zipf(
        self,
        num_requests      : int,
        deadline_range_ms : tuple,
    ) -> List[Request]:
        """
        Poisson inter-arrival + Zipf service selection.

        Inter-arrival time
        ------------------
        Each gap between successive requests is drawn from:

            delta_t ~ Exponential(1 / lambda)

        where lambda = POISSON_ARRIVAL_RATE (mean requests per ms).
        This models a memoryless Poisson arrival process.

        Service selection
        -----------------
        At the start of each call, services are ranked by their current
        EMA popularity to compute Zipf weights.  The ranking is recomputed
        every call so that popularity drift between episodes is reflected
        immediately in the service selection distribution.
        """
        services = list(self.catalogue.values())
        weights  = _zipf_weights(services, self.zipf_alpha)

        requests = []
        for _ in range(num_requests):
            # Exponential inter-arrival (Poisson process)
            self._sim_time += np.random.exponential(1.0 / self.arrival_rate)

            svc         = random.choices(services, weights=weights, k=1)[0]
            origin_node = random.choice(self.leaf_nodes)
            dlo, dhi    = _DEADLINE_RANGES.get(svc.criticality, deadline_range_ms)
            deadline    = random.uniform(dlo, dhi)

            requests.append(Request(
                service_id     = svc.service_id,
                origin_node_id = origin_node.node_id,
                deadline_ms    = deadline,
                arrival_time   = self._sim_time,
            ))
        return requests

    # ------------------------------------------------------------------
    # Uniform inter-arrival + popularity-weighted service selection
    # ------------------------------------------------------------------

    def _generate_uniform(
        self,
        num_requests      : int,
        deadline_range_ms : tuple,
    ) -> List[Request]:
        """
        Uniform inter-arrival + raw-popularity-weighted service selection.

        Preserved as a baseline / legacy model for ablation comparisons.
        Inter-arrival is Uniform(1, 5) ms; service selected by popularity.
        """
        services = list(self.catalogue.values())
        weights  = [s.popularity for s in services]

        requests = []
        for _ in range(num_requests):
            self._sim_time += random.uniform(1, 5)

            svc         = random.choices(services, weights=weights, k=1)[0]
            origin_node = random.choice(self.leaf_nodes)
            dlo, dhi    = _DEADLINE_RANGES.get(svc.criticality, deadline_range_ms)
            deadline    = random.uniform(dlo, dhi)

            requests.append(Request(
                service_id     = svc.service_id,
                origin_node_id = origin_node.node_id,
                deadline_ms    = deadline,
                arrival_time   = self._sim_time,
            ))
        return requests

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_zipf_distribution(self) -> Dict[int, float]:
        """
        Return the current Zipf probability for each service_id.
        Useful for logging / debugging demand distribution.
        Only meaningful when model == "poisson_zipf".
        """
        services = list(self.catalogue.values())
        weights  = _zipf_weights(services, self.zipf_alpha)
        return {svc.service_id: w for svc, w in zip(services, weights)}

    def expected_arrival_rate(self) -> float:
        """
        Expected number of requests per ms.
        For Poisson: lambda = POISSON_ARRIVAL_RATE.
        For uniform: 1 / E[Uniform(1,5)] = 1 / 3 ms.
        """
        if self.model == "poisson_zipf":
            return self.arrival_rate
        return 1.0 / 3.0

    @property
    def sim_time(self) -> float:
        """
        Current simulated clock in ms.

        Exposed so the orchestrator can compute actual_interval_ms:

            actual_interval_ms = sim_time(current) - sim_time(previous)

        This measured elapsed time replaces all fixed rate constants
        in the placement cost formula.
        """
        return self._sim_time

    def __repr__(self) -> str:
        return (
            f"RequestGenerator(model={self.model!r}, "
            f"services={len(self.catalogue)}, "
            f"leaf_nodes={len(self.leaf_nodes)}, "
            f"sim_time={self._sim_time:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# CSV request loader
# ---------------------------------------------------------------------------

def load_requests_from_csv(
    csv_path    : str,
    catalogue   : Dict[int, Service],
    node_map    : Dict[str, "EdgeNode"],
) -> List[Request]:
    """
    Load a pre-generated request trace from a CSV file.

    Expected columns (written by training/data/generate_requests.py):
        request_id, service_id, deadline_ms, origin_node_id,
        arrival_time_ms, cpu_cycles

    Only service_id, deadline_ms, origin_node_id, and arrival_time_ms
    are used to construct Request objects.  cpu_cycles is ignored here
    (consumed by the EDF scheduler internally via service.size).

    Any row whose service_id or origin_node_id does not exist in the
    provided catalogue / node_map is silently skipped so that a
    sub-catalogue inference run stays consistent.

    Parameters
    ----------
    csv_path  : absolute or relative path to the requests CSV
    catalogue : {service_id -> Service}  active service catalogue
    node_map  : {node_id   -> EdgeNode}  topology node lookup

    Returns
    -------
    List of Request objects sorted by arrival_time_ms (ascending)
    """
    import csv as _csv

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Request CSV not found: {csv_path}\n"
            "Generate one with:  python training/data/generate_requests.py"
        )

    import os as _os
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in _csv.DictReader(f):
            svc_id  = int(row["service_id"])
            node_id = row["origin_node_id"].strip()
            if svc_id not in catalogue:
                continue
            if node_id not in node_map:
                continue
            rows.append(Request(
                service_id     = svc_id,
                origin_node_id = node_id,
                deadline_ms    = float(row["deadline_ms"]),
                arrival_time   = float(row["arrival_time_ms"]),
            ))

    # Guarantee arrival-time order (CSV should already be sorted, but be safe)
    rows.sort(key=lambda r: r.arrival_time)
    return rows


import os

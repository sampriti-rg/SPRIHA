"""
simulation.py
-------------
Top-level simulation engine that ties together:
  - Network Topology (N-Ary hierarchical tree)
  - Service Catalogue + priority_init warm start
  - Request Generator  (Fast Process)
  - Edge Orchestrator / DDQN Agent  (Slow Process)

Episode loop
------------
  For each episode:
    1. [Fast]  Generate requests -> route -> collect outcomes
    2. [Slow]  Every PROVISIONING_INTERVAL requests -> orchestrator.provision()
               Final provision call of the episode passes episode_done=True
               so terminal transitions are correctly flagged in the buffer.
    3. [End]   orchestrator.end_episode() -> epsilon decay, target sync,
               metric recording
    4. Drift service popularity to simulate exogenous demand shifts
"""

import random
from tqdm import tqdm
from typing import Dict

from edge_caching.config import (
    NUM_EPISODES, REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL,
    NUM_SERVICES, NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP, RANDOM_SEED,
    TRAFFIC_MODEL,
)
from edge_caching.core.service import generate_service_catalogue
from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator


class Simulation:
    """
    Master simulation class.

    Parameters
    ----------
    num_levels    : depth of the N-Ary tree  (runtime configurable)
    num_children  : children per node         (runtime configurable)
    cache_cap     : cache slots per node      (runtime configurable)
    num_services  : size of service catalogue
    num_episodes  : training episodes
    req_per_ep    : requests per episode
    prov_interval : slow process fires every N requests
    seed          : reproducibility seed
    init_strategy : one-time cache pre-seeding strategy (runs before DDQN)
                    'priority' - rank by (omega * revenue), place leaf-first
                    'none'     - all edge caches start empty
    """

    def __init__(
        self,
        num_levels   : int = NUM_LEVELS,
        num_children : int = NUM_CHILDREN,
        cache_cap    : int = NODE_CACHE_CAP,
        num_services : int = NUM_SERVICES,
        num_episodes : int = NUM_EPISODES,
        req_per_ep   : int = REQUESTS_PER_EPISODE,
        prov_interval: int = PROVISIONING_INTERVAL,
        seed         : int = RANDOM_SEED,
        init_strategy: str = "priority",
        traffic_model: str = TRAFFIC_MODEL,
    ):
        self.num_episodes  = num_episodes
        self.req_per_ep    = req_per_ep
        self.prov_interval = prov_interval
        self.seed          = seed
        self.init_strategy = init_strategy
        self.traffic_model = traffic_model

        random.seed(seed)

        # Build service catalogue
        catalogue_list = generate_service_catalogue(num_services, seed=seed)
        self.catalogue : Dict = {s.service_id: s for s in catalogue_list}

        # Build network topology + cloud cache
        self.topology = NetworkTopology(
            num_levels=num_levels,
            num_children=num_children,
            cache_cap=cache_cap,
        )
        self.topology.prime_cloud_cache(self.catalogue)

        # One-time edge cache pre-seeding (before DDQN takes over)
        self.init_report = None
        if init_strategy == "priority":
            self.init_report = self.topology.priority_init(self.catalogue)

        # Request generator
        self.req_gen = RequestGenerator(
            catalogue  = self.catalogue,
            leaf_nodes = self.topology.leaf_nodes,
            seed       = seed,
            model      = traffic_model,
        )

        # Edge Orchestrator (centralised DDQN)
        self.orchestrator = EdgeOrchestrator(
            nodes     = self.topology.all_nodes,
            catalogue = self.catalogue,
        )

        print(self.topology.summary())
        print(f"Services      : {num_services}")
        print(f"Episodes      : {num_episodes}")
        print(f"Req/Episode   : {req_per_ep}")
        print(f"Prov Interval : {prov_interval}")
        print(f"Init Strategy : {init_strategy}")
        print(f"Traffic Model : {traffic_model}")
        if self.init_report:
            self.init_report.print_report()
        print()

    def run(self) -> EdgeOrchestrator:
        """
        Run the full multi-episode simulation.

        Returns
        -------
        orchestrator : EdgeOrchestrator
            Contains all recorded episode metrics for post-run analysis.
        """
        for episode in tqdm(range(self.num_episodes), desc="Training"):

            requests = self.req_gen.generate(self.req_per_ep)

            ep_reward     = 0.0
            ep_executed   = 0
            ep_forwarded  = 0
            ep_violations = 0
            ep_loss       = 0.0
            prov_calls    = 0

            # How many provision calls will fire this episode?
            total_prov_calls = self.req_per_ep // self.prov_interval

            for idx, req in enumerate(requests):

                # Fast Process: route the request
                origin_node = self.topology.node_map[req.origin_node_id]
                origin_node.receive_request(req, self.catalogue)

                # Tally outcomes from the completed request
                if req.status == "executed":
                    ep_executed += 1
                    # forwarded = executed somewhere other than origin
                    if req.executed_at_node != req.origin_node_id:
                        ep_forwarded += 1
                elif req.status == "dropped":
                    ep_violations += 1

                # Slow Process: provision every N requests
                if (idx + 1) % self.prov_interval == 0:
                    prov_calls += 1
                    is_last    = (prov_calls == total_prov_calls)
                    # Pass current simulated clock so the orchestrator can
                    # compute actual_interval_ms = sim_time - prev_sim_time.
                    # This replaces all fixed rate constants in placement cost.
                    metrics = self.orchestrator.provision(
                        episode_done = is_last,
                        sim_time_ms  = self.req_gen.sim_time,
                    )
                    ep_reward += metrics["interval_reward"]
                    ep_loss   += metrics["avg_loss"]

            # Drift service popularity between episodes
            for svc in self.catalogue.values():
                svc.drift_popularity()

            avg_loss = ep_loss / max(prov_calls, 1)

            # Episode boundary: epsilon decay + target sync + metric record
            self.orchestrator.end_episode(
                episode_reward   = ep_reward,
                total_executed   = ep_executed,
                total_requests   = self.req_per_ep,
                total_forwarded  = ep_forwarded,
                total_violations = ep_violations,
                avg_loss         = avg_loss,
            )

        print("\nSimulation complete.")
        return self.orchestrator

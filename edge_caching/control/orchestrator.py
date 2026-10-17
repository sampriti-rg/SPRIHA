"""
orchestrator.py
---------------
Edge Orchestrator: centralised controller that hosts the DDQN agent
and drives the Slow Process (service provisioning decisions).

Responsibilities
----------------
1. Collect global state snapshots from ALL edge nodes after each interval
2. Build the 8-element state vector for each (service, node) pair
3. Query the DDQN agent for INSERT / DELETE / KEEP decisions
4. Apply decisions back to nodes via apply_action()
5. Compute rewards and push (s, a, r, s', done) transitions to replay buffer
6. Trigger one gradient update per provision cycle (respecting LEARN_EVERY)
7. Decay epsilon and synchronise target network at episode boundaries
8. Update service popularity via EMA and reset per-interval counters

Provision cycle (called every PROVISIONING_INTERVAL requests)
-------------------------------------------------------------
    for each node:
        for each service:
            s      = build_state_vector(service, node_snapshot)
            a      = agent.select_action(s)              <- epsilon-greedy
            cost   = node.apply_action(a, service)       <- INSERT/DELETE/KEEP
            r      = compute_reward(deltas, cost, ...)
            s'     = build_state_vector(service, new_snapshot)
            agent.store_transition(s, a, r, s', done)    <- done on episode end
    agent.learn()                                        <- ONE gradient step
    service.update_popularity(EMA)
    service.reset_interval_counters()
    node.reset_interval_stats()

Episode boundary (called by Simulation.run() at the end of each episode)
-------------------------------------------------------------------------
    agent.decay_epsilon()                                <- exploration decay
    agent.sync_target()   (every TARGET_UPDATE episodes) <- hard sync
    agent.soft_update_target() is an alternative         <- soft Polyak sync
"""

from typing import Dict, List
import numpy as np

from edge_caching.core.node import EdgeNode
from edge_caching.core.service import Service
from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.agents.dqn_agent import DQNAgent
from edge_caching.config import TARGET_UPDATE ,GAMMA

# Registry used for the ablation study: swap --agent-type ddqn|dqn on the
# training CLI to change which RL algorithm drives service provisioning.
AGENT_REGISTRY = {
    "ddqn": DDQNAgent,
    "dqn" : DQNAgent,
}


STATE_DIM = 10  # size, popularity, omega, revenue, fwd_rate, exe_rate, cache_util, sla_viol, is_cached, slack_norm


class EdgeOrchestrator:
    """
    Central orchestrator hosting the DDQN agent.

    Parameters
    ----------
    nodes     : all edge nodes excluding the cloud root
    catalogue : {service_id -> Service}
    """

    def __init__(
        self,
        nodes     : List[EdgeNode],
        catalogue : Dict[int, Service],
        agent_type: str = "ddqn",
    ):
        self.nodes      = nodes
        self.catalogue  = catalogue
        self.num_nodes  = len(nodes)
        self.num_svcs   = len(catalogue)
        self.agent_type = agent_type.lower()

        agent_cls = AGENT_REGISTRY.get(self.agent_type)
        if agent_cls is None:
            raise ValueError(
                f"Unknown agent_type '{agent_type}'. "
                f"Valid options: {list(AGENT_REGISTRY.keys())}"
            )

        self.agent = agent_cls(
            state_dim    = STATE_DIM,
            num_nodes    = self.num_nodes,
            num_services = self.num_svcs,
        )

        # ---- Episode-level metric history ----------------------------
        self.episode_rewards    : List[float] = []
        self.episode_profits    : List[float] = []
        self.episode_hit_rates  : List[float] = []
        self.episode_fwd_counts : List[float] = []
        self.episode_violations : List[int]   = []
        self.episode_losses     : List[float] = []
        self.episode_overestimation_gap: List[float] = [] #new

        self._episode        : int             = 0
        self._prev_snapshots : Dict[str, dict] = {}
        self._interval_reward_sequence: List[float] = [] # NEW: raw per-interval rewards this episode

        # Simulated clock at the previous provision call.
        # actual_interval_ms = sim_time(current) - _last_provision_sim_time
        # Initialised to 0.0 so the first interval uses sim_time directly
        # as its full elapsed duration, which is correct.
        self._last_provision_sim_time : float = 0.0

    # ==================================================================
    # Slow-process provision cycle
    # ==================================================================

    def provision(
        self,
        episode_done : bool  = False,
        sim_time_ms  : float = 0.0,
    ) -> dict:
        """
        Run one full provisioning cycle.

        Parameters
        ----------
        episode_done : bool
            Pass True on the LAST provision call of an episode so that
            the terminal transitions are stored with done=True, correctly
            zeroing out the Bellman bootstrap for those transitions.

        Returns
        -------
        dict with interval-level metrics:
            interval_reward, placement_cost,
            executed_delta, forwarded_delta, violation_delta,
            avg_loss, epsilon
        """
        # ── Compute actual elapsed interval from simulated clock ───────
        # This is the real simulated ms between consecutive provision
        # cycles.  It varies naturally with Poisson inter-arrival
        # randomness — no config constant or assumption needed.
        actual_interval_ms = max(
            sim_time_ms - self._last_provision_sim_time,
            1.0,   # floor prevents division-by-zero on very first call
        )
        self._last_provision_sim_time = sim_time_ms

        # ---- Step 1: snapshot all nodes BEFORE applying actions ------
        current_snapshots: Dict[str, dict] = {
            node.node_id: node.get_stats_snapshot()
            for node in self.nodes
        }

        # Compute total user requests this interval for EMA popularity update.
        # Use service.demand_count (incremented once per user request at the
        # origin leaf node) NOT forwarded+executed counts across all nodes.
        # The forwarded+executed sum counts each hop separately, inflating
        # the denominator 3-6x and collapsing all popularity values to 0.05.
        total_requests_interval = max(
            sum(service.demand_count for service in self.catalogue.values()),
            1,
        )

        interval_reward       = 0.0
        interval_placement    = 0.0
        total_executed_delta  = 0
        total_forwarded_delta = 0

        # ---- Step 2: for each (node, service) pair -------------------
        # for node in self.nodes:
        #     snap = current_snapshots[node.node_id]
        #     prev = self._prev_snapshots.get(node.node_id, {})

        #     for svc_id, service in self.catalogue.items():

        #         # Build current state vector
        #         state_vec = DDQNAgent.build_state_vector(service, snap)

        #         # Select action (epsilon-greedy)
        #         action = self.agent.select_action(state_vec)

        #         # Apply action; pass measured interval for placement cost
        #         placement_cost = node.apply_action(
        #             action, service, actual_interval_ms
        #         )
        #         interval_placement += placement_cost

        #         # Compute deltas against previous interval snapshot
        #         prev_fwd = prev.get("forwarded_count", {}).get(svc_id, 0)
        #         prev_exe = prev.get("executed_count",  {}).get(svc_id, 0)
        #         prev_srv = prev.get("served_count",    {}).get(svc_id, 0)
        #         cur_fwd  = snap["forwarded_count"].get(svc_id, 0)
        #         cur_exe  = snap["executed_count"].get(svc_id, 0)
        #         cur_srv  = snap["served_count"].get(svc_id, 0)

        #         delta_exe = max(0, cur_exe - prev_exe)
        #         delta_fwd = max(0, cur_fwd - prev_fwd)
        #         delta_srv = max(0, cur_srv - prev_srv)

        #         total_executed_delta  += delta_exe
        #         total_forwarded_delta += delta_fwd

        #         # forward_term = LAMBDA_FORWARD * revenue * omega * delta_forwarded
        #         # Physical latency consequences (deadline burn) are handled by
        #         # subtract_hop_latency() in the simulation. No hop_cost needed.
        #         reward = DDQNAgent.compute_reward(
        #             service         = service,
        #             delta_served    = delta_srv,
        #             delta_executed  = delta_exe,
        #             delta_forwarded = delta_fwd,
        #             placement_cost  = placement_cost,
        #         )
        #         interval_reward += reward

        #         # Build next-state vector AFTER applying the action
        #         next_snap      = node.get_stats_snapshot()
        #         next_state_vec = DDQNAgent.build_state_vector(service, next_snap)

        #         # Store transition � done=True only on the last interval
        #         # of the episode so Bellman target is not bootstrapped
        #         self.agent.store_transition(
        #             state_vec, action, reward, next_state_vec,
        #             done=episode_done,
        #         )

        # ---- Step 2: VECTORIZED batch action selection (5-10x faster!) ---
        # Collect ALL state vectors in one batch
        # all_states = []
        # all_pairs = []  # Store (node, service, svc_id) to maintain order
        
        # for node in self.nodes:
        #     snap = current_snapshots[node.node_id]
        #     for svc_id, service in self.catalogue.items():
        #         # Build state vector
        #         state_vec = DDQNAgent.build_state_vector(service, snap)
        #         all_states.append(state_vec)
        #         all_pairs.append((node, service, svc_id))
        
        # # Convert to numpy array for batch processing
        # states_batch = np.array(all_states, dtype=np.float32)
        
        # # VECTORIZED: Get ALL actions in one GPU-accelerated batch forward pass
        # actions_batch = self.agent.select_actions_batch(states_batch)
        
        # # ---- Step 3: Apply actions (now using batch results) ---
        # for idx, (node, service, svc_id) in enumerate(all_pairs):
        #     action = int(actions_batch[idx])  # Get action from batch
        #     prev = self._prev_snapshots.get(node.node_id, {})
        #     snap = current_snapshots[node.node_id]
        #     state_vec = all_states[idx]  # Reuse already-computed state
            
        #     # Apply action; pass measured interval for placement cost
        #     placement_cost = node.apply_action(
        #         action, service, actual_interval_ms
        #     )
        #     interval_placement += placement_cost

        #     # Compute deltas against previous interval snapshot
        #     prev_fwd = prev.get("forwarded_count", {}).get(svc_id, 0)
        #     prev_exe = prev.get("executed_count",  {}).get(svc_id, 0)
        #     prev_srv = prev.get("served_count",    {}).get(svc_id, 0)
        #     cur_fwd  = snap["forwarded_count"].get(svc_id, 0)
        #     cur_exe  = snap["executed_count"].get(svc_id, 0)
        #     cur_srv  = snap["served_count"].get(svc_id, 0)

        #     delta_exe = max(0, cur_exe - prev_exe)
        #     delta_fwd = max(0, cur_fwd - prev_fwd)
        #     delta_srv = max(0, cur_srv - prev_srv)

        #     total_executed_delta  += delta_exe
        #     total_forwarded_delta += delta_fwd

        #     reward = DDQNAgent.compute_reward(
        #         service         = service,
        #         delta_served    = delta_srv,
        #         delta_executed  = delta_exe,
        #         delta_forwarded = delta_fwd,
        #         placement_cost  = placement_cost,
        #     )
        #     interval_reward += reward

        #     # Build next-state vector AFTER applying the action
        #     #next_snap      = node.get_stats_snapshot()
        #     #next_state_vec = DDQNAgent.build_state_vector(service, next_snap)
        #     #next_state_vec = DDQNAgent.build_state_vector(service, snap)
        #     next_state_vec = state_vec
        #     # Store transition
        #     self.agent.store_transition(
        #         state_vec, action, reward, next_state_vec,
        #         done=episode_done,
        #     )
        # ---- Step 2: VECTORIZED batch action selection ---
        # all_pairs = []
        # for node in self.nodes:
        #     snap = current_snapshots[node.node_id]
        #     prev = self._prev_snapshots.get(node.node_id, {})
        #     for svc_id, service in self.catalogue.items():
        #         all_pairs.append((node, service, svc_id, snap, prev))
        
        # # Build all states
        # all_states = [
        #     DDQNAgent.build_state_vector(p[1], p[3])  # service, snap
        #     for p in all_pairs
        # ]
        
        # # Batch action selection
        # states_batch = np.array(all_states, dtype=np.float32)
        # actions_batch = self.agent.select_actions_batch(states_batch)
        
        # # ---- Step 3: Apply actions ---
        # for idx, (node, service, svc_id, snap, prev) in enumerate(all_pairs):
        #     action = int(actions_batch[idx])
        #     state_vec = all_states[idx]
            

        #     before_action_snap = node.get_stats_snapshot()
        #     before_fwd = before_action_snap["forwarded_count"].get(svc_id, 0)
        #     before_exe = before_action_snap["executed_count"].get(svc_id, 0)
        #     before_srv = before_action_snap["served_count"].get(svc_id, 0)
  
        #     placement_cost = node.apply_action(action, service, actual_interval_ms)
        #     interval_placement += placement_cost


        #     after_action_snap = node.get_stats_snapshot()
        #     after_fwd = after_action_snap["forwarded_count"].get(svc_id, 0)
        #     after_exe = after_action_snap["executed_count"].get(svc_id, 0)
        #     after_srv = after_action_snap["served_count"].get(svc_id, 0)


        #     # CAPTURE NEXT STATE AFTER ACTION
        #     next_snap = node.get_stats_snapshot()  # Get updated state

        #     prev_fwd = prev.get("forwarded_count", {}).get(svc_id, 0)
        #     prev_exe = prev.get("executed_count",  {}).get(svc_id, 0)
        #     prev_srv = prev.get("served_count",    {}).get(svc_id, 0)
        #     cur_fwd  = next_snap["forwarded_count"].get(svc_id, 0)
        #     cur_exe  = next_snap["executed_count"].get(svc_id, 0)
        #     cur_srv  = next_snap["served_count"].get(svc_id, 0)

        #     delta_exe = max(0, after_exe - before_exe)
        #     delta_fwd = max(0, after_fwd - before_fwd)
        #     delta_srv = max(0, after_srv - before_srv)

        #     total_executed_delta  += delta_exe
        #     total_forwarded_delta += delta_fwd

           
        #     next_state_vec = DDQNAgent.build_state_vector(service, after_action_snap)


        #     reward = DDQNAgent.compute_reward(
        #         service, delta_srv, delta_exe, delta_fwd, placement_cost
        #     )
        #     interval_reward += reward

        #     self.agent.store_transition(
        #         state_vec, action, reward, next_state_vec, done=episode_done
        #     )

         # ---- Step 2: Sequential action selection with live state updates ---
        for node in self.nodes:
            snap = node.get_stats_snapshot()  # Fresh snapshot per node
            prev = self._prev_snapshots.get(node.node_id, {})
            
            for svc_id, service in self.catalogue.items():
                # Build state vector from CURRENT node state
                state_vec = DDQNAgent.build_state_vector(service, snap)
                
                # Select action (epsilon-greedy)
                action = self.agent.select_action(state_vec)
                
                # Apply action
                placement_cost = node.apply_action(action, service, actual_interval_ms)
                interval_placement += placement_cost
                
                prev_fwd = prev.get("forwarded_count", {}).get(svc_id, 0)
                prev_exe = prev.get("executed_count",  {}).get(svc_id, 0)
                prev_srv = prev.get("served_count",    {}).get(svc_id, 0)


                cur_fwd  = snap["forwarded_count"].get(svc_id, 0)
                cur_exe  = snap["executed_count"].get(svc_id, 0)
                cur_srv  = snap["served_count"].get(svc_id, 0)
           

                # Deltas measure only this action's effect
                delta_exe = max(0, cur_exe - prev_exe)
                delta_fwd = max(0, cur_fwd - prev_fwd)
                delta_srv = max(0, cur_srv - prev_srv)
                
                total_executed_delta += delta_exe
                total_forwarded_delta += delta_fwd
                
                # Compute reward
                reward = DDQNAgent.compute_reward(
                    service         = service,
                    delta_served    = delta_srv,
                    delta_executed  = delta_exe,
                    delta_forwarded = delta_fwd,
                    placement_cost  = placement_cost,
                )
                interval_reward += reward

             
                # Build next-state vector AFTER applying the action
                next_snap      = node.get_stats_snapshot()
                next_state_vec = DDQNAgent.build_state_vector(service, next_snap)
                
                
                
                # Store transition
                self.agent.store_transition(
                    state_vec, action, reward, next_state_vec, done=episode_done
                )
                
              
        # ---- Step 3: ONE gradient update per provision cycle ---------
        # learn() internally enforces WARMUP_STEPS and LEARN_EVERY guards,
        # so calling it unconditionally here is safe and correct.
        loss_val = self.agent.learn()

        # ---- Step 4: update service demand model ---------------------
        for service in self.catalogue.values():
            service.update_popularity(total_requests_interval)
            service.reset_interval_counters()

        # ---- Step 5: reset node interval stats ----------------------
        for node in self.nodes:
            node.reset_interval_stats()

        # ---- Step 6: persist snapshot for next cycle ----------------
        self._prev_snapshots = current_snapshots
        self._interval_reward_sequence.append(interval_reward)
        return {
            "interval_reward" : interval_reward,
            "placement_cost"  : interval_placement,
            "executed_delta"  : total_executed_delta,
            "forwarded_delta" : total_forwarded_delta,
            "avg_loss"        : loss_val,
            "epsilon"         : self.agent.epsilon,
        }

    # ==================================================================
    # Episode boundary
    # ==================================================================

    def end_episode(
        self,
        episode_reward   : float,
        total_executed   : int,
        total_requests   : int,
        total_forwarded  : int,
        total_violations : int,
        avg_loss         : float,
    ) -> None:
        """
        Record end-of-episode metrics and advance the training schedule.

        Called once per episode by Simulation.run() AFTER the last
        provision() call for that episode.

        Actions taken here (NOT inside provision())
        -------------------------------------------
        1. epsilon decay  : reduce exploration rate by EPSILON_DECAY
        2. target sync    : hard-copy online -> target every TARGET_UPDATE
                            episodes, keeping the regression target stable
                            for TARGET_UPDATE complete training rounds.

        Separating these from provision() ensures they run exactly once
        per episode regardless of how many provision intervals fire.
        """
        hit_rate = total_executed / max(total_requests, 1)
        profit   = sum(n.total_revenue for n in self.nodes)

        self.episode_rewards.append(episode_reward)
        self.episode_profits.append(profit)
        self.episode_hit_rates.append(hit_rate)
        self.episode_fwd_counts.append(total_forwarded)
        self.episode_violations.append(total_violations)
        self.episode_losses.append(avg_loss)

        # ---- Overestimation-gap bookkeeping (NEW) --------------------
        actual_return = sum(
            (GAMMA ** i) * r for i, r in enumerate(self._interval_reward_sequence)
        )
        predicted_q_hist = getattr(self.agent, "predicted_q_history", [])
        predicted_q_avg = (
            sum(predicted_q_hist) / len(predicted_q_hist)
        ) if predicted_q_hist else 0.0
        self.episode_overestimation_gap.append(predicted_q_avg - actual_return)


        # Reset per-episode node revenue accumulators
        for node in self.nodes:
            node.total_revenue = 0.0

        # Epsilon decay: once per episode
        self.agent.decay_epsilon()

        # Target network synchronisation
        # _disable_hard_sync is set to True by finetunehighload.py to prevent
        # Bellman instability during near-greedy high-load refinement.
        if getattr(self, "_disable_hard_sync", False):
            # Soft Polyak update: theta_t <- TAU*theta_online + (1-TAU)*theta_t
            # Moves the target 0.5% per episode -- smooth, no abrupt jumps.
            self.agent.soft_update_target()
        else:
            # Hard sync every TARGET_UPDATE episodes (normal training mode)
            if (self._episode + 1) % TARGET_UPDATE == 0:
                self.agent.sync_target()

        self._episode += 1

    # ==================================================================
    # Checkpoint delegation
    # ==================================================================

    def save_checkpoint(self, path: str) -> None:
        """Delegate checkpoint save to the DDQN agent."""
        self.agent.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Delegate checkpoint load to the DDQN agent."""
        self.agent.load_checkpoint(path)

    def start_episode(self) -> None:
        """
        Call ONCE at the start of each episode (before the first provision()
        call) to reset per-episode tracking used for overestimation-bias
        measurement (predicted Q-value vs actual discounted return).
        """
        self._interval_reward_sequence = []
        self.agent.reset_episode_tracking()
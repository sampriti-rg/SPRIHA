"""
dqn_agent.py
------------
Vanilla (single) Deep Q-Network agent — used as an ABLATION baseline against
the project's default DDQNAgent.

Why this file exists
---------------------
The main framework trains a Double DQN (DDQN) agent (see ddqn_agent.py),
which de-couples *action selection* (online network) from *action
evaluation* (target network) to reduce Q-value overestimation bias.

For the ablation study we want to measure what happens when that
"Double Q-learning" component is REMOVED, i.e. we replace DDQN with the
classic DQN (Mnih et al., 2015) update rule:

    DDQN target:  y = r + gamma * Q_target( s', argmax_a' Q_online(s', a') )
    DQN  target:  y = r + gamma * max_a' Q_target(s', a')

Everything else (network architecture, PER replay buffer, GPU handling,
checkpointing, epsilon schedule, target sync) is inherited unchanged from
DDQNAgent so that the ONLY varying factor in the ablation is the Bellman
target computation. This isolates the effect of Double Q-learning.
"""

import torch
import torch.nn as nn

from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.config import GAMMA, GRAD_CLIP_NORM


class DQNAgent(DDQNAgent):
    """
    Single (vanilla) DQN agent.

    Identical to DDQNAgent in every respect except the `learn()` method,
    which uses the standard DQN target:

        y = r + gamma * max_a' Q_target(s', a')

    instead of the Double-DQN target used by the parent class.
    """
    def __init__(self, state_dim: int, num_nodes: int, num_services: int):
        super().__init__(state_dim, num_nodes, num_services)
        print("[DQNAgent] Initialized as standard DQN (shared target net for selection + evaluation)")

    def learn(self) -> float:
        if not self.memory.is_ready():
            return 0.0

        self._learn_counter += 1
        if self._learn_counter % 1 != 0:  # kept for symmetry w/ LEARN_EVERY guard below
            pass
        from edge_caching.config import LEARN_EVERY, BATCH_SIZE
        if self._learn_counter % LEARN_EVERY != 0:
            return 0.0

        states_np, actions_np, rewards_np, next_states_np, dones_np, weights, indices = \
            self.memory.sample(BATCH_SIZE)

        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        if self.device.type == 'cuda':
            from torch.amp import autocast
            with autocast('cuda'):
                current_q = self.online_net(states).gather(1, actions).squeeze(1)

                # --- Vanilla DQN target (no Double-Q decoupling) -------
                with torch.no_grad():
                    target_q_vals = self.target_net(next_states).max(dim=1)[0]
                    target_q = rewards + GAMMA * target_q_vals * (1.0 - dones)

                td_errors = (current_q - target_q).detach().float().cpu().numpy()
                elementwise_loss = nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')
                loss = (weights_t * elementwise_loss).mean()
        else:
            current_q = self.online_net(states).gather(1, actions).squeeze(1)

            # --- Vanilla DQN target (no Double-Q decoupling) -----------
            with torch.no_grad():
                target_q_vals = self.target_net(next_states).max(dim=1)[0]
                target_q = rewards + GAMMA * target_q_vals * (1.0 - dones)

            td_errors = (current_q - target_q).detach().cpu().numpy()
            elementwise_loss = nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (weights_t * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

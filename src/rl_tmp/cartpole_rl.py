from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .cartpole_system import rk4_step_torch

@dataclass
class CartPolePPOResult:
    model: nn.Module
    history: dict[str, list[float]]

class CartPoleActorCritic(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


def _initial_batch(batch_size, device):
    state = torch.zeros(batch_size, 4, device=device)
    state[:, 0] = 0.1 * torch.randn(batch_size, device=device)
    state[:, 1] = 0.12 * torch.randn(batch_size, device=device)
    state[:, 2:] = 0.05 * torch.randn(batch_size, 2, device=device)
    return state


def train_cartpole_ppo(config, updates=80, batch_size=128, horizon=128, dt=0.02, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = CartPoleActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    gamma, gae_lambda, clip_ratio = 0.99, 0.95, 0.2
    state = _initial_batch(batch_size, device)
    history = {"mean_reward": [], "failure_rate": []}

    for _ in range(updates):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        dones = []

        for _step in range(horizon):
            with torch.no_grad():
                logits, value = model(state)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)

            force = torch.where(
                action == 0,
                torch.full_like(action, -config.max_force, dtype=torch.float32),
                torch.full_like(action, config.max_force, dtype=torch.float32),
            )
            next_state = rk4_step_torch(state, force, dt, config)
            done = (next_state[:, 0].abs() > 2.4) | (next_state[:, 1].abs() > 0.5)
            reward = (
                1.0
                - 0.05 * next_state[:, 0].square()
                - 0.25 * next_state[:, 1].square()
                - 0.002 * next_state[:, 2].square()
                - 0.005 * next_state[:, 3].square()
                - 5.0 * done.float()
            )

            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            state = next_state
            if done.any():
                state[done] = _initial_batch(int(done.sum()), device)

        with torch.no_grad():
            _, next_value = model(state)

        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        old_log_probs_t = torch.stack(old_log_probs)
        rewards_t = torch.stack(rewards)
        values_t = torch.stack(values)
        dones_t = torch.stack(dones)

        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros(batch_size, device=device)
        bootstrap = next_value
        for step in range(horizon - 1, -1, -1):
            nonterminal = (~dones_t[step]).float()
            delta = rewards_t[step] + gamma * bootstrap * nonterminal - values_t[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step] = gae
            bootstrap = values_t[step]

        returns = advantages + values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_states = states_t.reshape(-1, 4)
        flat_actions = actions_t.flatten()
        flat_old_log_probs = old_log_probs_t.flatten()
        flat_advantages = advantages.flatten()
        flat_returns = returns.flatten()

        for _epoch in range(4):
            permutation = torch.randperm(len(flat_states), device=device)
            for start in range(0, len(flat_states), 1024):
                index = permutation[start:start + 1024]
                logits, value = model(flat_states[index])
                distribution = Categorical(logits=logits)
                new_log_prob = distribution.log_prob(flat_actions[index])
                ratio = (new_log_prob - flat_old_log_probs[index]).exp()
                unclipped = ratio * flat_advantages[index]
                clipped = (
                    torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                    * flat_advantages[index]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (value - flat_returns[index]).square().mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        history["mean_reward"].append(float(rewards_t.mean().cpu()))
        history["failure_rate"].append(float(dones_t.float().mean().cpu()))

    return CartPolePPOResult(model=model, history=history)


@torch.no_grad()
def rollout_cartpole_policy(model, initial_state, config, t_final=5.0, dt=0.02, device="cpu"):
    state = torch.tensor(initial_state, dtype=torch.float32, device=device)[None]
    states = [state[0].cpu().numpy().copy()]
    forces = []
    success = True

    for _ in range(int(round(t_final / dt))):
        logits, _ = model(state)
        action = logits.argmax(dim=-1)
        force = torch.where(
            action == 0,
            torch.full_like(action, -config.max_force, dtype=torch.float32),
            torch.full_like(action, config.max_force, dtype=torch.float32),
        )
        state = rk4_step_torch(state, force, dt, config)
        states.append(state[0].cpu().numpy().copy())
        forces.append(float(force.item()))

        if abs(float(state[0, 0])) > 2.4 or abs(float(state[0, 1])) > 0.5:
            success = False
            break

    return np.asarray(states), np.asarray(forces), success

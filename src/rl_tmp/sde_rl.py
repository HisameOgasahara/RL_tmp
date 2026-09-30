from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.distributions import Normal

from .sde_system import (
    SDEOscillatorConfig,
    controlled_sde_drift,
    sde_diffusion,
)


@dataclass
class SDERLResult:
    policy: nn.Module
    history: dict[str, list[float]]


class GaussianSDEPolicy(nn.Module):
    """pi_theta: R^2 -> probability distributions on R."""

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 2):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.mean_network = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.tensor(-0.5))

    def distribution(self, state: torch.Tensor) -> Normal:
        mean = self.mean_network(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def deterministic_action(
        self,
        state: torch.Tensor,
        max_action: float,
    ) -> torch.Tensor:
        return max_action * torch.tanh(self.mean_network(state))


def _discounted_returns(
    rewards: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """rewards: [time, batch] -> returns: [time, batch]."""
    returns = torch.zeros_like(rewards)
    running = torch.zeros(
        rewards.shape[1],
        dtype=rewards.dtype,
        device=rewards.device,
    )

    for index in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running

    return returns


def train_sde_reinforce(
    initial_state: torch.Tensor,
    goal_state: torch.Tensor,
    config: SDEOscillatorConfig,
    t_final: float,
    dt: float = 0.05,
    episodes: int = 8000,
    batch_size: int = 32,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    action_cost: float = 0.01,
    max_action: float = 3.0,
    gradient_clip: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
) -> SDERLResult:
    torch.manual_seed(seed)

    policy = GaussianSDEPolicy(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)

    initial_state = initial_state.to(device)
    goal_state = goal_state.to(device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
    )

    horizon = max(1, int(round(t_final / dt)))
    updates = max(1, math.ceil(episodes / batch_size))

    history = {"mean_episode_return": []}

    best_return = -float("inf")
    best_state_dict = deepcopy(policy.state_dict())

    for update in range(updates):
        current_batch_size = min(
            batch_size,
            episodes - update * batch_size,
        )

        state = initial_state.unsqueeze(0).repeat(
            current_batch_size,
            1,
        )

        log_probs = []
        rewards = []

        for step in range(horizon):
            distribution = policy.distribution(state)
            raw_action = distribution.sample()
            action = max_action * torch.tanh(raw_action)

            log_prob = distribution.log_prob(raw_action).sum(dim=-1)

            t = torch.tensor(
                step * dt,
                dtype=state.dtype,
                device=device,
            )

            drift = controlled_sde_drift(
                t=t,
                state=state,
                action=action,
                config=config,
            )
            diffusion = sde_diffusion(
                t=t,
                state=state,
                config=config,
            )
            noise = torch.randn_like(state)

            next_state = (
                state
                + drift * dt
                + diffusion * math.sqrt(dt) * noise
            )

            state_cost = (
                next_state - goal_state.unsqueeze(0)
            ).square().sum(dim=-1)

            control_cost = (
                action_cost
                * action.square().sum(dim=-1)
            )

            reward = -(state_cost + control_cost)

            log_probs.append(log_prob)
            rewards.append(reward.detach())
            state = next_state.detach()

        log_prob_tensor = torch.stack(log_probs)
        reward_tensor = torch.stack(rewards)

        returns = _discounted_returns(
            rewards=reward_tensor,
            gamma=gamma,
        )

        baseline = returns.mean(
            dim=1,
            keepdim=True,
        )

        advantages = returns - baseline
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        loss = -(
            log_prob_tensor * advantages
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=gradient_clip,
        )
        optimizer.step()

        with torch.no_grad():
            policy.log_std.clamp_(
                min=-2.5,
                max=0.5,
            )

        mean_episode_return = (
            reward_tensor.sum(dim=0).mean()
        ).item()

        history["mean_episode_return"].append(
            mean_episode_return
        )

        if mean_episode_return > best_return:
            best_return = mean_episode_return
            best_state_dict = deepcopy(
                policy.state_dict()
            )

    policy.load_state_dict(best_state_dict)

    return SDERLResult(
        policy=policy,
        history=history,
    )


@torch.no_grad()
def rollout_sde_policy(
    policy: GaussianSDEPolicy,
    initial_state: torch.Tensor,
    config: SDEOscillatorConfig,
    t_final: float,
    dt: float,
    max_action: float,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    state = initial_state.to(device)
    horizon = max(1, int(round(t_final / dt)))

    times = [0.0]
    states = [state.cpu()]
    actions = []

    for step in range(horizon):
        action = policy.deterministic_action(
            state,
            max_action=max_action,
        )

        t = torch.tensor(
            step * dt,
            dtype=state.dtype,
            device=device,
        )

        drift = controlled_sde_drift(
            t=t,
            state=state,
            action=action,
            config=config,
        )
        diffusion = sde_diffusion(
            t=t,
            state=state,
            config=config,
        )
        noise = torch.randn_like(state)

        state = (
            state
            + drift * dt
            + diffusion * math.sqrt(dt) * noise
        )

        actions.append(action.cpu())
        states.append(state.cpu())
        times.append((step + 1) * dt)

    return (
        torch.tensor(times),
        torch.stack(states),
        torch.stack(actions),
    )

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from .solver import rk4_step
from .system import OscillatorConfig, controlled_oscillator_field


@dataclass
class RLResult:
    policy: nn.Module
    history: dict[str, list[float]]


class GaussianPolicy(nn.Module):
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


def discounted_returns(
    rewards: list[torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    returns = []
    running = torch.zeros_like(rewards[0])

    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)

    returns.reverse()
    values = torch.stack(returns)
    return (values - values.mean()) / (values.std() + 1e-8)


def train_reinforce(
    initial_state: torch.Tensor,
    goal_state: torch.Tensor,
    config: OscillatorConfig,
    t_final: float,
    dt: float = 0.05,
    episodes: int = 1000,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    action_cost: float = 0.01,
    max_action: float = 3.0,
    seed: int = 0,
    device: str = "cpu",
) -> RLResult:
    torch.manual_seed(seed)

    policy = GaussianPolicy(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)

    initial_state = initial_state.to(device)
    goal_state = goal_state.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    horizon = max(1, int(round(t_final / dt)))

    history = {"episode_return": []}

    for _ in range(episodes):
        state = initial_state.clone()
        log_probs: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []

        for step in range(horizon):
            distribution = policy.distribution(state)
            raw_action = distribution.sample()
            action = max_action * torch.tanh(raw_action)

            log_prob = distribution.log_prob(raw_action).sum()

            t = torch.tensor(step * dt, device=device)
            dt_tensor = torch.tensor(dt, device=device)

            def field(
                current_time: torch.Tensor,
                current_state: torch.Tensor,
            ) -> torch.Tensor:
                return controlled_oscillator_field(
                    current_time,
                    current_state,
                    action,
                    config,
                )

            next_state = rk4_step(field, t, state, dt_tensor)

            state_cost = (next_state - goal_state).square().sum()
            control_cost = action_cost * action.square().sum()
            reward = -(state_cost + control_cost)

            log_probs.append(log_prob)
            rewards.append(reward.detach())
            state = next_state.detach()

        returns = discounted_returns(rewards, gamma)
        log_prob_tensor = torch.stack(log_probs)
        loss = -(log_prob_tensor * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = torch.stack(rewards).sum()
        history["episode_return"].append(float(episode_return.cpu()))

    return RLResult(policy=policy, history=history)


@torch.no_grad()
def rollout_policy(
    policy: GaussianPolicy,
    initial_state: torch.Tensor,
    config: OscillatorConfig,
    t_final: float,
    dt: float,
    max_action: float,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = initial_state.to(device)
    horizon = max(1, int(round(t_final / dt)))

    times = [0.0]
    states = [state.cpu()]
    actions = []

    for step in range(horizon):
        action = policy.deterministic_action(state, max_action=max_action)

        t = torch.tensor(step * dt, device=device)
        dt_tensor = torch.tensor(dt, device=device)

        def field(
            current_time: torch.Tensor,
            current_state: torch.Tensor,
        ) -> torch.Tensor:
            return controlled_oscillator_field(
                current_time,
                current_state,
                action,
                config,
            )

        state = rk4_step(field, t, state, dt_tensor)

        actions.append(action.cpu())
        states.append(state.cpu())
        times.append((step + 1) * dt)

    return (
        torch.tensor(times),
        torch.stack(states),
        torch.stack(actions),
    )

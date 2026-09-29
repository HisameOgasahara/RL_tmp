from dataclasses import dataclass

import torch
from torch import nn

from .sde_solver import euler_maruyama_integrate


@dataclass
class NeuralSDEResult:
    model: nn.Module
    history: dict[str, list[float]]


class LearnedSDE(nn.Module):
    """Learned drift b_theta: R^2 -> R^2 and diagonal diffusion sigma_theta: R^2 -> R^2."""

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 2):
        super().__init__()

        drift_layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        diffusion_layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]

        for _ in range(hidden_layers - 1):
            drift_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
            diffusion_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])

        drift_layers.append(nn.Linear(hidden_dim, 2))
        diffusion_layers.extend([nn.Linear(hidden_dim, 2), nn.Softplus()])

        self.drift_network = nn.Sequential(*drift_layers)
        self.diffusion_network = nn.Sequential(*diffusion_layers)

    def drift(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time
        return self.drift_network(state)

    def diffusion(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time
        return self.diffusion_network(state) + 1e-4


def train_neural_sde(
    initial_states: torch.Tensor,
    times: torch.Tensor,
    target_trajectories: torch.Tensor,
    noise_increments: torch.Tensor,
    steps: int = 1200,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> NeuralSDEResult:
    torch.manual_seed(seed)

    model = LearnedSDE(hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    initial_states = initial_states.to(device)
    times = times.to(device)
    target_trajectories = target_trajectories.to(device)
    noise_increments = noise_increments.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"trajectory_mse": []}

    for _ in range(steps):
        prediction = euler_maruyama_integrate(
            drift=model.drift,
            diffusion=model.diffusion,
            initial_state=initial_states,
            times=times,
            noise_increments=noise_increments,
        )
        loss = (prediction - target_trajectories).square().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["trajectory_mse"].append(float(loss.detach().cpu()))

    return NeuralSDEResult(model=model, history=history)


@torch.no_grad()
def predict_neural_sde(
    model: LearnedSDE,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    noise_increments: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    return euler_maruyama_integrate(
        drift=model.drift,
        diffusion=model.diffusion,
        initial_state=initial_state.to(device),
        times=times.to(device),
        noise_increments=noise_increments.to(device),
    ).cpu()

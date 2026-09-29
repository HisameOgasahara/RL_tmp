from dataclasses import dataclass

import torch
from torch import nn
from torchdiffeq import odeint


@dataclass
class NeuralODEResult:
    model: nn.Module
    history: dict[str, list[float]]


class LearnedVectorField(nn.Module):
    """f_theta: R^2 -> R^2."""

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 2):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        del time
        return self.network(state)


def train_neural_ode(
    initial_states: torch.Tensor,
    times: torch.Tensor,
    target_trajectories: torch.Tensor,
    steps: int = 1000,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> NeuralODEResult:
    torch.manual_seed(seed)

    model = LearnedVectorField(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)

    initial_states = initial_states.to(device)
    times = times.to(device)
    target_trajectories = target_trajectories.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"trajectory_mse": []}

    for _ in range(steps):
        prediction = odeint(
            model,
            initial_states,
            times,
            method="rk4",
        )

        loss = (prediction - target_trajectories).square().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["trajectory_mse"].append(float(loss.detach().cpu()))

    return NeuralODEResult(model=model, history=history)


@torch.no_grad()
def predict_neural_ode(
    model: LearnedVectorField,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    trajectory = odeint(
        model,
        initial_state.to(device),
        times.to(device),
        method="rk4",
    )
    return trajectory.cpu()


@torch.no_grad()
def evaluate_learned_field(
    model: LearnedVectorField,
    states: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    dummy_time = torch.tensor(0.0, device=device)
    values = model(dummy_time, states.to(device))
    return values.cpu()

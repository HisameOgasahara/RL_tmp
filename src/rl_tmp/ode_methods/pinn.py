from dataclasses import dataclass

import torch
from torch import nn

from .system import OscillatorConfig


@dataclass
class PINNResult:
    model: nn.Module
    history: dict[str, list[float]]


class TrajectoryPINN(nn.Module):
    """x_theta: R -> R^2."""

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 2):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        return self.network(time)


def pinn_losses(
    model: TrajectoryPINN,
    collocation_times: torch.Tensor,
    initial_state: torch.Tensor,
    config: OscillatorConfig,
    initial_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    times = collocation_times.detach().clone().requires_grad_(True)
    state = model(times)

    q = state[:, 0:1]
    v = state[:, 1:2]

    dq_dt = torch.autograd.grad(
        q,
        times,
        grad_outputs=torch.ones_like(q),
        create_graph=True,
    )[0]
    dv_dt = torch.autograd.grad(
        v,
        times,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
    )[0]

    residual_q = dq_dt - v
    residual_v = dv_dt + config.spring * q + config.damping * v
    residual_loss = (residual_q.square() + residual_v.square()).mean()

    t0 = torch.zeros((1, 1), dtype=times.dtype, device=times.device)
    initial_prediction = model(t0).squeeze(0)
    initial_loss = (initial_prediction - initial_state).square().mean()

    total_loss = residual_loss + initial_weight * initial_loss
    return total_loss, residual_loss, initial_loss


def train_pinn(
    initial_state: torch.Tensor,
    config: OscillatorConfig,
    t_final: float,
    steps: int = 2000,
    collocation_points: int = 128,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    learning_rate: float = 1e-3,
    initial_weight: float = 10.0,
    seed: int = 0,
    device: str = "cpu",
) -> PINNResult:
    torch.manual_seed(seed)

    model = TrajectoryPINN(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)

    initial_state = initial_state.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "total": [],
        "residual": [],
        "initial": [],
    }

    for _ in range(steps):
        collocation_times = torch.rand(
            collocation_points,
            1,
            device=device,
        ) * t_final

        total_loss, residual_loss, initial_loss = pinn_losses(
            model=model,
            collocation_times=collocation_times,
            initial_state=initial_state,
            config=config,
            initial_weight=initial_weight,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history["total"].append(float(total_loss.detach().cpu()))
        history["residual"].append(float(residual_loss.detach().cpu()))
        history["initial"].append(float(initial_loss.detach().cpu()))

    return PINNResult(model=model, history=history)


@torch.no_grad()
def predict_pinn(
    model: TrajectoryPINN,
    times: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    prediction = model(times.to(device))
    return prediction.cpu()

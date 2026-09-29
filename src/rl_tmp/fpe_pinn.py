from dataclasses import dataclass

import torch
from torch import nn

from .sde_system import SDEOscillatorConfig


@dataclass
class FPEPINNResult:
    model: nn.Module
    history: dict[str, list[float]]


class DensityPINN(nn.Module):
    """rho_theta: R x R^2 -> R_{>0}."""

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 3):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)
        self.positive = nn.Softplus()

    def forward(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        values = torch.cat((time, state), dim=-1)
        return self.positive(self.network(values)) + 1e-8


def gaussian_density(
    state: torch.Tensor,
    mean: torch.Tensor,
    std: float,
) -> torch.Tensor:
    variance = std**2
    difference = state - mean
    exponent = -0.5 * difference.square().sum(dim=-1, keepdim=True) / variance
    normalizer = 2.0 * torch.pi * variance
    return torch.exp(exponent) / normalizer


def fpe_residual(
    model: DensityPINN,
    time: torch.Tensor,
    state: torch.Tensor,
    config: SDEOscillatorConfig,
) -> torch.Tensor:
    time = time.detach().clone().requires_grad_(True)
    state = state.detach().clone().requires_grad_(True)

    rho = model(time, state)
    rho_t = torch.autograd.grad(
        rho,
        time,
        grad_outputs=torch.ones_like(rho),
        create_graph=True,
    )[0]

    rho_grad = torch.autograd.grad(
        rho,
        state,
        grad_outputs=torch.ones_like(rho),
        create_graph=True,
    )[0]

    q = state[:, 0:1]
    v = state[:, 1:2]
    drift_q = v
    drift_v = -config.spring * q - config.damping * v

    flux_q = drift_q * rho
    flux_v = drift_v * rho

    flux_q_grad = torch.autograd.grad(
        flux_q,
        state,
        grad_outputs=torch.ones_like(flux_q),
        create_graph=True,
    )[0][:, 0:1]
    flux_v_grad = torch.autograd.grad(
        flux_v,
        state,
        grad_outputs=torch.ones_like(flux_v),
        create_graph=True,
    )[0][:, 1:2]

    rho_v = rho_grad[:, 1:2]
    rho_vv = torch.autograd.grad(
        rho_v,
        state,
        grad_outputs=torch.ones_like(rho_v),
        create_graph=True,
    )[0][:, 1:2]

    return rho_t + flux_q_grad + flux_v_grad - 0.5 * config.noise**2 * rho_vv


def train_fpe_pinn(
    initial_mean: torch.Tensor,
    initial_std: float,
    config: SDEOscillatorConfig,
    t_final: float,
    q_range: tuple[float, float] = (-3.0, 3.0),
    v_range: tuple[float, float] = (-3.0, 3.0),
    steps: int = 2500,
    collocation_points: int = 256,
    initial_points: int = 256,
    hidden_dim: int = 64,
    hidden_layers: int = 3,
    learning_rate: float = 1e-3,
    initial_weight: float = 10.0,
    seed: int = 0,
    device: str = "cpu",
) -> FPEPINNResult:
    torch.manual_seed(seed)

    model = DensityPINN(hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    initial_mean = initial_mean.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "fpe": [], "initial": []}

    for _ in range(steps):
        time = torch.rand(collocation_points, 1, device=device) * t_final
        q = q_range[0] + (q_range[1] - q_range[0]) * torch.rand(collocation_points, 1, device=device)
        v = v_range[0] + (v_range[1] - v_range[0]) * torch.rand(collocation_points, 1, device=device)
        state = torch.cat((q, v), dim=-1)

        residual = fpe_residual(model=model, time=time, state=state, config=config)
        fpe_loss = residual.square().mean()

        initial_q = q_range[0] + (q_range[1] - q_range[0]) * torch.rand(initial_points, 1, device=device)
        initial_v = v_range[0] + (v_range[1] - v_range[0]) * torch.rand(initial_points, 1, device=device)
        initial_state = torch.cat((initial_q, initial_v), dim=-1)
        initial_time = torch.zeros(initial_points, 1, device=device)

        target_density = gaussian_density(initial_state, initial_mean, initial_std)
        predicted_density = model(initial_time, initial_state)
        initial_loss = (predicted_density - target_density).square().mean()

        total_loss = fpe_loss + initial_weight * initial_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history["total"].append(float(total_loss.detach().cpu()))
        history["fpe"].append(float(fpe_loss.detach().cpu()))
        history["initial"].append(float(initial_loss.detach().cpu()))

    return FPEPINNResult(model=model, history=history)

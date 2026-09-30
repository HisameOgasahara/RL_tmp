from dataclasses import dataclass
import math

import torch
from torch import nn

from .sde_system import SDEOscillatorConfig


@dataclass
class FPEPINNResult:
    model: nn.Module
    history: dict[str, list[float]]


class DensityPINN(nn.Module):
    """Gaussian density rho_theta: R x R^2 -> R_{>0} with exact initial moments."""

    def __init__(
        self,
        initial_mean: torch.Tensor,
        initial_std: float,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 5))

        self.network = nn.Sequential(*layers)
        self.register_buffer("initial_mean", initial_mean.detach().clone().float())
        self.initial_std = float(initial_std)
        self.base_diagonal = math.log(math.expm1(self.initial_std))

    def moments(
        self,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if time.ndim == 1:
            time = time.unsqueeze(-1)

        raw = self.network(time)

        mean = self.initial_mean.unsqueeze(0) + time * raw[:, 0:2]

        diagonal_1 = torch.nn.functional.softplus(
            self.base_diagonal + time[:, 0] * raw[:, 2]
        ) + 1e-4
        lower_off_diagonal = time[:, 0] * raw[:, 3]
        diagonal_2 = torch.nn.functional.softplus(
            self.base_diagonal + time[:, 0] * raw[:, 4]
        ) + 1e-4

        cholesky = torch.zeros(
            (time.shape[0], 2, 2),
            dtype=time.dtype,
            device=time.device,
        )
        cholesky[:, 0, 0] = diagonal_1
        cholesky[:, 1, 0] = lower_off_diagonal
        cholesky[:, 1, 1] = diagonal_2

        covariance = cholesky @ cholesky.transpose(-1, -2)
        return mean, covariance

    def forward(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)

        mean, covariance = self.moments(time)
        difference = state - mean
        inverse_covariance = torch.linalg.inv(covariance)

        quadratic = torch.einsum(
            "bi,bij,bj->b",
            difference,
            inverse_covariance,
            difference,
        ).unsqueeze(-1)

        determinant = torch.linalg.det(covariance).unsqueeze(-1)
        normalizer = 2.0 * torch.pi * torch.sqrt(determinant)

        return torch.exp(-0.5 * quadratic) / normalizer


def fpe_moment_residuals(
    model: DensityPINN,
    times: torch.Tensor,
    config: SDEOscillatorConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For this linear SDE with Gaussian initial density, the Fokker-Planck PDE
    is equivalent to ODEs for mean mu(t) and covariance Sigma(t):

        dmu/dt = A mu
        dSigma/dt = A Sigma + Sigma A^T + B B^T
    """
    times = times.detach().clone().requires_grad_(True)
    mean, covariance = model.moments(times)

    mean_derivative_components = []
    for index in range(2):
        component = mean[:, index:index + 1]
        derivative = torch.autograd.grad(
            component,
            times,
            grad_outputs=torch.ones_like(component),
            create_graph=True,
        )[0]
        mean_derivative_components.append(derivative[:, 0])

    mean_derivative = torch.stack(mean_derivative_components, dim=-1)

    covariance_derivative_rows = []
    for row in range(2):
        row_derivatives = []
        for column in range(2):
            component = covariance[:, row, column:column + 1]
            derivative = torch.autograd.grad(
                component,
                times,
                grad_outputs=torch.ones_like(component),
                create_graph=True,
            )[0]
            row_derivatives.append(derivative[:, 0])

        covariance_derivative_rows.append(
            torch.stack(row_derivatives, dim=-1)
        )

    covariance_derivative = torch.stack(
        covariance_derivative_rows,
        dim=1,
    )

    drift_matrix = torch.tensor(
        [
            [0.0, 1.0],
            [-config.spring, -config.damping],
        ],
        dtype=times.dtype,
        device=times.device,
    )

    diffusion_covariance = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, config.noise**2],
        ],
        dtype=times.dtype,
        device=times.device,
    )

    target_mean_derivative = mean @ drift_matrix.transpose(0, 1)

    target_covariance_derivative = (
        drift_matrix.unsqueeze(0) @ covariance
        + covariance @ drift_matrix.transpose(0, 1).unsqueeze(0)
        + diffusion_covariance.unsqueeze(0)
    )

    mean_residual = mean_derivative - target_mean_derivative
    covariance_residual = (
        covariance_derivative - target_covariance_derivative
    )

    return mean_residual, covariance_residual


def train_fpe_pinn(
    initial_mean: torch.Tensor,
    initial_std: float,
    config: SDEOscillatorConfig,
    t_final: float,
    q_range: tuple[float, float] = (-3.0, 3.0),
    v_range: tuple[float, float] = (-3.0, 3.0),
    steps: int = 2500,
    collocation_points: int = 128,
    initial_points: int = 256,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    learning_rate: float = 1e-3,
    initial_weight: float = 10.0,
    seed: int = 0,
    device: str = "cpu",
) -> FPEPINNResult:
    del q_range
    del v_range
    del initial_points
    del initial_weight

    torch.manual_seed(seed)

    model = DensityPINN(
        initial_mean=initial_mean,
        initial_std=initial_std,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "total": [],
        "mean_residual": [],
        "covariance_residual": [],
    }

    for _ in range(steps):
        times = torch.rand(
            collocation_points,
            1,
            device=device,
        ) * t_final

        mean_residual, covariance_residual = fpe_moment_residuals(
            model=model,
            times=times,
            config=config,
        )

        mean_loss = mean_residual.square().mean()
        covariance_loss = covariance_residual.square().mean()
        total_loss = mean_loss + covariance_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        optimizer.step()

        history["total"].append(float(total_loss.detach().cpu()))
        history["mean_residual"].append(float(mean_loss.detach().cpu()))
        history["covariance_residual"].append(
            float(covariance_loss.detach().cpu())
        )

    return FPEPINNResult(
        model=model,
        history=history,
    )

from dataclasses import dataclass

import torch
from torch import nn

from .cartpole_system import cartpole_field_torch


@dataclass
class CartPolePINNResult:
    model: nn.Module
    history: dict[str, list[float]]


class CartPoleTrajectoryPINN(nn.Module):
    """x_theta: R -> R^4 with the initial condition built into the map."""

    def __init__(
        self,
        initial_state,
        t_final: float,
        hidden_dim: int = 96,
    ):
        super().__init__()

        self.register_buffer(
            "initial_state",
            torch.as_tensor(initial_state, dtype=torch.float32),
        )
        self.t_final = float(t_final)

        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)

        normalized_time = time / self.t_final

        return (
            self.initial_state
            + normalized_time * self.network(normalized_time)
        )


def _residual_loss(
    model: CartPoleTrajectoryPINN,
    times: torch.Tensor,
    config,
    lqr_gain: torch.Tensor,
) -> torch.Tensor:
    state = model(times)

    derivatives = []

    for component in range(4):
        derivative = torch.autograd.grad(
            state[:, component:component + 1],
            times,
            grad_outputs=torch.ones_like(
                state[:, component:component + 1]
            ),
            create_graph=True,
        )[0]

        derivatives.append(derivative)

    state_derivative = torch.cat(derivatives, dim=1)

    force = torch.clamp(
        -(state @ lqr_gain.T),
        -config.max_force,
        config.max_force,
    )

    physics_derivative = cartpole_field_torch(
        state,
        force,
        config,
    )

    return (
        state_derivative - physics_derivative
    ).square().mean()


def train_cartpole_pinn(
    initial_state,
    config,
    lqr_gain,
    t_final: float = 1.0,
    steps: int = 3000,
    collocation_points: int = 160,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> CartPolePINNResult:
    torch.manual_seed(seed)

    model = CartPoleTrajectoryPINN(
        initial_state=initial_state,
        t_final=t_final,
    ).to(device)

    lqr_gain = torch.tensor(
        lqr_gain,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "residual": [],
    }

    for _ in range(steps):
        random_times = (
            torch.rand(
                collocation_points - 2,
                1,
                device=device,
            )
            * t_final
        )

        boundary_times = torch.tensor(
            [[0.0], [t_final]],
            device=device,
        )

        times = torch.cat(
            (boundary_times, random_times),
            dim=0,
        ).requires_grad_(True)

        loss = _residual_loss(
            model=model,
            times=times,
            config=config,
            lqr_gain=lqr_gain,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["residual"].append(
            float(loss.detach().cpu())
        )

    return CartPolePINNResult(
        model=model,
        history=history,
    )


@torch.no_grad()
def predict_cartpole_pinn(
    model: CartPoleTrajectoryPINN,
    times,
    device: str = "cpu",
) -> torch.Tensor:
    times = torch.tensor(
        times,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)

    return model(times).cpu()

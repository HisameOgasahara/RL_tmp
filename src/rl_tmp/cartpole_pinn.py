from dataclasses import dataclass
import torch
from torch import nn
from .cartpole_system import cartpole_field_torch

@dataclass
class CartPolePINNResult:
    model: nn.Module
    history: dict[str, list[float]]

class CartPoleTrajectoryPINN(nn.Module):
    def __init__(self, initial_state, t_final, hidden_dim=96):
        super().__init__()
        self.register_buffer("initial_state", torch.as_tensor(initial_state, dtype=torch.float32))
        self.t_final = float(t_final)
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )
    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        tau = t / self.t_final
        return self.initial_state + tau * self.network(tau)


def _residual_loss(model, t, config, K):
    state = model(t)
    derivatives = []
    for component in range(4):
        derivative = torch.autograd.grad(
            state[:, component:component+1], t,
            grad_outputs=torch.ones_like(state[:, component:component+1]),
            create_graph=True,
        )[0]
        derivatives.append(derivative)
    dx_dt = torch.cat(derivatives, dim=1)
    force = torch.clamp(-(state @ K.T), -config.max_force, config.max_force)
    physics = cartpole_field_torch(state, force, config)
    return (dx_dt - physics).square().mean()


def train_cartpole_pinn(initial_state, config, lqr_gain, t_final=2.0, steps=2200, collocation_points=160, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = CartPoleTrajectoryPINN(initial_state, t_final).to(device)
    K = torch.tensor(lqr_gain, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"residual": []}
    for _ in range(steps):
        random_t = torch.rand(collocation_points - 2, 1, device=device) * t_final
        boundary_t = torch.tensor([[0.0], [t_final]], device=device)
        t = torch.cat((boundary_t, random_t), dim=0).requires_grad_(True)
        loss = _residual_loss(model, t, config, K)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history["residual"].append(float(loss.detach().cpu()))
    return CartPolePINNResult(model=model, history=history)

@torch.no_grad()
def predict_cartpole_pinn(model, times, device="cpu"):
    t = torch.tensor(times, dtype=torch.float32, device=device)[:, None]
    return model(t).cpu()

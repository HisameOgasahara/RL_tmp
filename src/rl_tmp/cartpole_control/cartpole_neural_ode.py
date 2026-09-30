from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .cartpole_solver import solve_cartpole_reference

@dataclass
class NeuralODECartPoleResult:
    model: nn.Module
    history: dict[str, list[float]]

class CartPoleVectorField(nn.Module):
    def __init__(self, hidden_dim=96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )
    def forward(self, state):
        return self.network(state)


def make_solver_dataset(config, trajectories=48, points=41, t_final=1.5, seed=0):
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, t_final, points)
    states = []
    derivatives = []
    from .cartpole_system import cartpole_field_numpy
    for _ in range(trajectories):
        initial = np.array([
            rng.uniform(-0.4, 0.4),
            rng.uniform(-0.22, 0.22),
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.3, 0.3),
        ])
        trajectory = solve_cartpole_reference(initial, times, config)
        states.append(trajectory)
        derivatives.append(np.stack([cartpole_field_numpy(s, 0.0, config) for s in trajectory]))
    return (
        torch.tensor(np.concatenate(states), dtype=torch.float32),
        torch.tensor(np.concatenate(derivatives), dtype=torch.float32),
    )


def train_cartpole_neural_ode(config, steps=1200, learning_rate=2e-3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    states, targets = make_solver_dataset(config, seed=seed)
    states, targets = states.to(device), targets.to(device)
    model = CartPoleVectorField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"field_mse": []}
    for _ in range(steps):
        index = torch.randint(0, len(states), (min(512, len(states)),), device=device)
        prediction = model(states[index])
        loss = (prediction - targets[index]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history["field_mse"].append(float(loss.detach().cpu()))
    return NeuralODECartPoleResult(model=model, history=history)

@torch.no_grad()
def rollout_learned_field(model, initial_state, times, device="cpu"):
    state = torch.tensor(initial_state, dtype=torch.float32, device=device)
    states = [state.cpu()]
    for i in range(len(times) - 1):
        dt = float(times[i + 1] - times[i])
        def step_field(s): return model(s)
        k1 = step_field(state)
        k2 = step_field(state + 0.5 * dt * k1)
        k3 = step_field(state + 0.5 * dt * k2)
        k4 = step_field(state + dt * k3)
        state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        states.append(state.cpu())
    return torch.stack(states)

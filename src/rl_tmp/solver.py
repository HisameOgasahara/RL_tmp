from collections.abc import Callable

import torch

from .system import OscillatorConfig, oscillator_field


TensorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rk4_step(
    field: TensorField,
    t: torch.Tensor,
    state: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    k1 = field(t, state)
    k2 = field(t + dt / 2, state + dt * k1 / 2)
    k3 = field(t + dt / 2, state + dt * k2 / 2)
    k4 = field(t + dt, state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk4_integrate(
    field: TensorField,
    initial_state: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    states = [initial_state]
    state = initial_state

    for index in range(len(times) - 1):
        t = times[index]
        dt = times[index + 1] - times[index]
        state = rk4_step(field, t, state, dt)
        states.append(state)

    return torch.stack(states, dim=0)


def solve_reference(
    initial_state: torch.Tensor,
    times: torch.Tensor,
    config: OscillatorConfig,
) -> torch.Tensor:
    def field(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return oscillator_field(t, state, config)

    return rk4_integrate(field, initial_state, times)


def make_reference_dataset(
    initial_states: torch.Tensor,
    times: torch.Tensor,
    config: OscillatorConfig,
) -> torch.Tensor:
    """Return shape [time, batch, 2]."""
    return solve_reference(initial_states, times, config)

from dataclasses import dataclass

import torch


@dataclass
class OscillatorConfig:
    spring: float = 1.0
    damping: float = 0.3


def oscillator_field(
    t: torch.Tensor,
    state: torch.Tensor,
    config: OscillatorConfig,
) -> torch.Tensor:
    """f: R x R^2 -> R^2 for the damped harmonic oscillator."""
    del t
    q = state[..., 0]
    v = state[..., 1]

    dq = v
    dv = -config.spring * q - config.damping * v

    return torch.stack((dq, dv), dim=-1)


def controlled_oscillator_field(
    t: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
    config: OscillatorConfig,
) -> torch.Tensor:
    """f_u: R x R^2 x R -> R^2 with control applied to acceleration."""
    base = oscillator_field(t, state, config)
    controlled = base.clone()
    controlled[..., 1] = controlled[..., 1] + action.squeeze(-1)
    return controlled

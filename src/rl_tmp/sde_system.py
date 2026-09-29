from dataclasses import dataclass

import torch


@dataclass
class SDEOscillatorConfig:
    spring: float = 1.0
    damping: float = 0.3
    noise: float = 0.35


def sde_drift(
    t: torch.Tensor,
    state: torch.Tensor,
    config: SDEOscillatorConfig,
) -> torch.Tensor:
    """b: R x R^2 -> R^2 for a stochastic damped oscillator."""
    del t
    q = state[..., 0]
    v = state[..., 1]

    dq = v
    dv = -config.spring * q - config.damping * v

    return torch.stack((dq, dv), dim=-1)


def sde_diffusion(
    t: torch.Tensor,
    state: torch.Tensor,
    config: SDEOscillatorConfig,
) -> torch.Tensor:
    """sigma: R x R^2 -> R^2 with noise applied to velocity."""
    del t
    diffusion = torch.zeros_like(state)
    diffusion[..., 1] = config.noise
    return diffusion


def controlled_sde_drift(
    t: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
    config: SDEOscillatorConfig,
) -> torch.Tensor:
    """b_u: R x R^2 x R -> R^2 with control applied to acceleration."""
    drift = sde_drift(t, state, config).clone()
    drift[..., 1] = drift[..., 1] + action.squeeze(-1)
    return drift

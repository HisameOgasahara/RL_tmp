from collections.abc import Callable

import torch

from .sde_system import SDEOscillatorConfig, sde_diffusion, sde_drift


TensorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def euler_maruyama_step(
    drift: TensorField,
    diffusion: TensorField,
    t: torch.Tensor,
    state: torch.Tensor,
    dt: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    return state + drift(t, state) * dt + diffusion(t, state) * torch.sqrt(dt) * noise


def euler_maruyama_integrate(
    drift: TensorField,
    diffusion: TensorField,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    noise_increments: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    states = [initial_state]
    state = initial_state

    for index in range(len(times) - 1):
        t = times[index]
        dt = times[index + 1] - times[index]

        if noise_increments is None:
            noise = torch.randn(
                state.shape,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
        else:
            noise = noise_increments[index]

        state = euler_maruyama_step(
            drift=drift,
            diffusion=diffusion,
            t=t,
            state=state,
            dt=dt,
            noise=noise,
        )
        states.append(state)

    return torch.stack(states, dim=0)


def solve_sde_reference(
    initial_state: torch.Tensor,
    times: torch.Tensor,
    config: SDEOscillatorConfig,
    noise_increments: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    def drift(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return sde_drift(t, state, config)

    def diffusion(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return sde_diffusion(t, state, config)

    return euler_maruyama_integrate(
        drift=drift,
        diffusion=diffusion,
        initial_state=initial_state,
        times=times,
        noise_increments=noise_increments,
        generator=generator,
    )


def make_sde_reference_dataset(
    initial_states: torch.Tensor,
    times: torch.Tensor,
    config: SDEOscillatorConfig,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=initial_states.device)
    generator.manual_seed(seed)

    noise_increments = torch.randn(
        (len(times) - 1, *initial_states.shape),
        dtype=initial_states.dtype,
        device=initial_states.device,
        generator=generator,
    )

    trajectories = solve_sde_reference(
        initial_state=initial_states,
        times=times,
        config=config,
        noise_increments=noise_increments,
    )
    return trajectories, noise_increments

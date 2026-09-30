from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .cartpole_system import rk4_step_torch


@dataclass
class CartPoleRLResult:
    policy: nn.Module
    history: dict[str, list[float]]


class LinearStateFeedbackPolicy(nn.Module):
    """pi_w: R^4 -> R for goal-conditioned Cart-Pole control."""

    def __init__(
        self,
        weights: torch.Tensor,
        goal_state: torch.Tensor,
    ):
        super().__init__()

        self.register_buffer(
            "weights",
            weights.detach().clone(),
        )

        self.register_buffer(
            "goal_state",
            goal_state.detach().clone(),
        )

    def forward(
        self,
        state: torch.Tensor,
        max_force: float,
    ) -> torch.Tensor:
        error = state - self.goal_state

        force = -(
            error * self.weights
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return torch.clamp(
            force,
            -max_force,
            max_force,
        )


def _evaluate_candidates(
    candidate_weights: torch.Tensor,
    initial_state: torch.Tensor,
    goal_state: torch.Tensor,
    config,
    t_final: float,
    dt: float,
    action_cost: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    candidate_count = candidate_weights.shape[0]

    state = (
        initial_state
        .unsqueeze(0)
        .repeat(candidate_count, 1)
    )

    horizon = max(
        1,
        int(round(t_final / dt)),
    )

    state_weights = torch.tensor(
        [1.0, 15.0, 0.2, 0.5],
        dtype=state.dtype,
        device=state.device,
    )

    total_cost = torch.zeros(
        candidate_count,
        dtype=state.dtype,
        device=state.device,
    )

    survived = torch.ones(
        candidate_count,
        dtype=torch.bool,
        device=state.device,
    )

    for _ in range(horizon):
        error = state - goal_state

        force = -(
            error * candidate_weights
        ).sum(dim=-1)

        force = torch.clamp(
            force,
            -config.max_force,
            config.max_force,
        )

        state = rk4_step_torch(
            state=state,
            force=force,
            dt=dt,
            config=config,
        )

        error = state - goal_state

        state_cost = (
            state_weights
            * error.square()
        ).sum(dim=-1)

        control_cost = (
            action_cost
            * force.square()
        )

        failed = (
            (state[:, 0].abs() > 2.4)
            | (state[:, 1].abs() > 0.5)
        )

        total_cost = (
            total_cost
            + state_cost
            + control_cost
            + 50.0 * failed.float()
        )

        survived = survived & (~failed)

    final_distance = (
        torch.linalg.vector_norm(
            state - goal_state,
            dim=-1,
        )
    )

    total_cost = (
        total_cost
        + 20.0 * final_distance
    )

    return (
        total_cost,
        state,
        final_distance,
        survived,
    )


def train_cartpole_policy_search(
    initial_state,
    goal_state,
    config,
    t_final: float = 5.0,
    dt: float = 0.02,
    iterations: int = 30,
    population_size: int = 512,
    elite_fraction: float = 0.1,
    initial_std: float = 20.0,
    minimum_std: float = 0.3,
    action_cost: float = 0.002,
    seed: int = 0,
    device: str = "cpu",
) -> CartPoleRLResult:
    torch.manual_seed(seed)

    initial_state = torch.as_tensor(
        initial_state,
        dtype=torch.float32,
        device=device,
    )

    goal_state = torch.as_tensor(
        goal_state,
        dtype=torch.float32,
        device=device,
    )

    mean = torch.zeros(
        4,
        dtype=torch.float32,
        device=device,
    )

    std = torch.full(
        (4,),
        initial_std,
        dtype=torch.float32,
        device=device,
    )

    elite_count = max(
        1,
        int(round(
            population_size
            * elite_fraction
        )),
    )

    history = {
        "elite_mean_cost": [],
        "elite_mean_goal_distance": [],
        "elite_survival_rate": [],
    }

    for _ in range(iterations):
        candidate_weights = (
            mean
            + std
            * torch.randn(
                population_size,
                4,
                device=device,
            )
        )

        (
            total_cost,
            _final_state,
            final_distance,
            survived,
        ) = _evaluate_candidates(
            candidate_weights=candidate_weights,
            initial_state=initial_state,
            goal_state=goal_state,
            config=config,
            t_final=t_final,
            dt=dt,
            action_cost=action_cost,
        )

        elite_indices = torch.topk(
            total_cost,
            k=elite_count,
            largest=False,
        ).indices

        elite_weights = (
            candidate_weights[
                elite_indices
            ]
        )

        mean = elite_weights.mean(
            dim=0
        )

        std = elite_weights.std(
            dim=0
        ).clamp_min(
            minimum_std
        )

        history[
            "elite_mean_cost"
        ].append(
            float(
                total_cost[
                    elite_indices
                ].mean().cpu()
            )
        )

        history[
            "elite_mean_goal_distance"
        ].append(
            float(
                final_distance[
                    elite_indices
                ].mean().cpu()
            )
        )

        history[
            "elite_survival_rate"
        ].append(
            float(
                survived[
                    elite_indices
                ].float().mean().cpu()
            )
        )

    policy = LinearStateFeedbackPolicy(
        weights=mean,
        goal_state=goal_state,
    ).to(device)

    return CartPoleRLResult(
        policy=policy,
        history=history,
    )


@torch.no_grad()
def rollout_cartpole_policy(
    policy: LinearStateFeedbackPolicy,
    initial_state,
    goal_state,
    config,
    t_final: float = 5.0,
    dt: float = 0.02,
    goal_tolerance: float = 0.1,
    device: str = "cpu",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    bool,
]:
    state = torch.as_tensor(
        initial_state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    goal_state = torch.as_tensor(
        goal_state,
        dtype=torch.float32,
        device=device,
    )

    horizon = max(
        1,
        int(round(t_final / dt)),
    )

    times = [0.0]
    states = [
        state[0].cpu().clone()
    ]
    actions = []

    for step in range(horizon):
        force = policy(
            state,
            max_force=config.max_force,
        )

        state = rk4_step_torch(
            state=state,
            force=force,
            dt=dt,
            config=config,
        )

        states.append(
            state[0].cpu().clone()
        )

        actions.append(
            force[0].cpu().clone()
        )

        times.append(
            (step + 1) * dt
        )

    states = torch.stack(states)
    actions = torch.stack(actions)

    final_distance = float(
        torch.linalg.vector_norm(
            states[-1]
            - goal_state.cpu()
        )
    )

    reached_goal = (
        final_distance
        <= goal_tolerance
    )

    return (
        torch.tensor(times),
        states,
        actions,
        final_distance,
        reached_goal,
    )

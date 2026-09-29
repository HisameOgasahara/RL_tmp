import matplotlib.pyplot as plt
import numpy as np
import torch

from .system import OscillatorConfig, oscillator_field


def _state_grid(
    q_range: tuple[float, float],
    v_range: tuple[float, float],
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    q = np.linspace(*q_range, grid_points)
    v = np.linspace(*v_range, grid_points)
    q_grid, v_grid = np.meshgrid(q, v)

    states = torch.tensor(
        np.stack((q_grid, v_grid), axis=-1),
        dtype=torch.float32,
    )
    return q_grid, v_grid, states


def plot_vector_field(
    config: OscillatorConfig,
    q_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    grid_points: int = 21,
    trajectory: torch.Tensor | None = None,
    title: str = "Vector field",
) -> None:
    q_grid, v_grid, states = _state_grid(q_range, v_range, grid_points)

    field = oscillator_field(
        torch.tensor(0.0),
        states,
        config,
    ).numpy()

    plt.figure(figsize=(7, 6))
    plt.quiver(
        q_grid,
        v_grid,
        field[..., 0],
        field[..., 1],
        alpha=0.7,
    )

    if trajectory is not None:
        values = trajectory.detach().cpu().numpy()
        plt.plot(values[..., 0], values[..., 1], linewidth=2)
        plt.scatter(values[0, 0], values[0, 1], s=50)

    plt.xlabel("q")
    plt.ylabel("v")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.show()


def plot_pinn_phase_field(
    config: OscillatorConfig,
    reference: torch.Tensor,
    pinn_trajectory: torch.Tensor,
    q_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    grid_points: int = 21,
) -> None:
    q_grid, v_grid, states = _state_grid(q_range, v_range, grid_points)

    field = oscillator_field(
        torch.tensor(0.0),
        states,
        config,
    ).numpy()

    reference_np = reference.detach().cpu().numpy()
    pinn_np = pinn_trajectory.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.quiver(
        q_grid,
        v_grid,
        field[..., 0],
        field[..., 1],
        alpha=0.6,
    )
    plt.plot(
        reference_np[:, 0],
        reference_np[:, 1],
        label="RK4 reference",
        linewidth=2,
    )
    plt.plot(
        pinn_np[:, 0],
        pinn_np[:, 1],
        "--",
        label="PINN trajectory",
        linewidth=2,
    )
    plt.scatter(reference_np[0, 0], reference_np[0, 1], s=50, label="initial")
    plt.xlabel("q")
    plt.ylabel("v")
    plt.title("True vector field + PINN trajectory")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def plot_loss(
    history: dict[str, list[float]],
    title: str,
    log_scale: bool = False,
) -> None:
    plt.figure(figsize=(7, 4))

    for name, values in history.items():
        plt.plot(values, label=name)

    if log_scale:
        plt.yscale("log")

    plt.xlabel("training step")
    plt.ylabel("metric")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def plot_trajectory_comparison(
    times: torch.Tensor,
    reference: torch.Tensor,
    prediction: torch.Tensor,
    title: str,
) -> None:
    times_np = times.detach().cpu().numpy()
    reference_np = reference.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(times_np, reference_np[:, 0], label="reference q")
    plt.plot(times_np, prediction_np[:, 0], "--", label="predicted q")
    plt.plot(times_np, reference_np[:, 1], label="reference v")
    plt.plot(times_np, prediction_np[:, 1], "--", label="predicted v")
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def plot_neural_ode_field_trajectory(
    true_config: OscillatorConfig,
    learned_model: torch.nn.Module,
    reference: torch.Tensor,
    learned_trajectory: torch.Tensor,
    q_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    grid_points: int = 17,
    device: str = "cpu",
) -> None:
    q_grid, v_grid, states = _state_grid(q_range, v_range, grid_points)

    true_field = oscillator_field(
        torch.tensor(0.0),
        states,
        true_config,
    )

    with torch.no_grad():
        learned_field = learned_model(
            torch.tensor(0.0, device=device),
            states.to(device),
        ).cpu()

    true_np = true_field.numpy()
    learned_np = learned_field.numpy()
    reference_np = reference.detach().cpu().numpy()
    learned_trajectory_np = learned_trajectory.detach().cpu().numpy()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].quiver(
        q_grid,
        v_grid,
        true_np[..., 0],
        true_np[..., 1],
        alpha=0.7,
    )
    axes[0].plot(
        reference_np[:, 0],
        reference_np[:, 1],
        label="RK4 reference",
        linewidth=2,
    )
    axes[0].set_title("True vector field + RK4 trajectory")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("v")
    axes[0].legend()

    axes[1].quiver(
        q_grid,
        v_grid,
        learned_np[..., 0],
        learned_np[..., 1],
        alpha=0.7,
    )
    axes[1].plot(
        learned_trajectory_np[:, 0],
        learned_trajectory_np[:, 1],
        label="Neural ODE trajectory",
        linewidth=2,
    )
    axes[1].set_title("Learned vector field + Neural ODE trajectory")
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("v")
    axes[1].legend()

    for axis in axes:
        axis.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_rl_closed_loop_field(
    policy: torch.nn.Module,
    config: OscillatorConfig,
    uncontrolled: torch.Tensor,
    controlled: torch.Tensor,
    goal_state: torch.Tensor,
    max_action: float,
    q_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    grid_points: int = 17,
    device: str = "cpu",
) -> None:
    q_grid, v_grid, states = _state_grid(q_range, v_range, grid_points)

    uncontrolled_field = oscillator_field(
        torch.tensor(0.0),
        states,
        config,
    )

    flat_states = states.reshape(-1, 2).to(device)
    with torch.no_grad():
        actions = policy.deterministic_action(
            flat_states,
            max_action=max_action,
        ).cpu()

    controlled_field = oscillator_field(
        torch.tensor(0.0),
        states,
        config,
    )
    controlled_field[..., 1] = (
        controlled_field[..., 1]
        + actions.reshape(grid_points, grid_points)
    )

    uncontrolled_np = uncontrolled.detach().cpu().numpy()
    controlled_np = controlled.detach().cpu().numpy()
    goal_np = goal_state.detach().cpu().numpy()
    uncontrolled_field_np = uncontrolled_field.numpy()
    controlled_field_np = controlled_field.numpy()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].quiver(
        q_grid,
        v_grid,
        uncontrolled_field_np[..., 0],
        uncontrolled_field_np[..., 1],
        alpha=0.7,
    )
    axes[0].plot(
        uncontrolled_np[:, 0],
        uncontrolled_np[:, 1],
        label="uncontrolled trajectory",
        linewidth=2,
    )
    axes[0].scatter(goal_np[0], goal_np[1], marker="*", s=140, label="goal")
    axes[0].set_title("Uncontrolled vector field")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("v")
    axes[0].legend()

    axes[1].quiver(
        q_grid,
        v_grid,
        controlled_field_np[..., 0],
        controlled_field_np[..., 1],
        alpha=0.7,
    )
    axes[1].plot(
        controlled_np[:, 0],
        controlled_np[:, 1],
        label="RL controlled trajectory",
        linewidth=2,
    )
    axes[1].scatter(goal_np[0], goal_np[1], marker="*", s=140, label="goal")
    axes[1].set_title("Policy closed-loop vector field")
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("v")
    axes[1].legend()

    for axis in axes:
        axis.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_rl_rollout(
    uncontrolled: torch.Tensor,
    controlled: torch.Tensor,
    goal_state: torch.Tensor,
    actions: torch.Tensor,
    times: torch.Tensor,
) -> None:
    uncontrolled_np = uncontrolled.detach().cpu().numpy()
    controlled_np = controlled.detach().cpu().numpy()
    goal_np = goal_state.detach().cpu().numpy()
    actions_np = actions.detach().cpu().numpy().squeeze(-1)
    times_np = times.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.plot(
        uncontrolled_np[:, 0],
        uncontrolled_np[:, 1],
        label="uncontrolled",
    )
    plt.plot(
        controlled_np[:, 0],
        controlled_np[:, 1],
        label="RL controlled",
    )
    plt.scatter(goal_np[0], goal_np[1], marker="*", s=140, label="goal")
    plt.xlabel("q")
    plt.ylabel("v")
    plt.title("RL control in phase space")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

    plt.figure(figsize=(7, 3))
    plt.plot(times_np[:-1], actions_np)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Policy action")
    plt.grid(alpha=0.2)
    plt.show()

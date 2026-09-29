import matplotlib.pyplot as plt
import numpy as np
import torch

from .system import OscillatorConfig, oscillator_field


def plot_vector_field(
    config: OscillatorConfig,
    q_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    grid_points: int = 21,
    trajectory: torch.Tensor | None = None,
    title: str = "Vector field",
) -> None:
    q = np.linspace(*q_range, grid_points)
    v = np.linspace(*v_range, grid_points)
    q_grid, v_grid = np.meshgrid(q, v)

    states = torch.tensor(
        np.stack((q_grid, v_grid), axis=-1),
        dtype=torch.float32,
    )
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


def plot_field_comparison(
    true_config: OscillatorConfig,
    learned_model: torch.nn.Module,
    q_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    grid_points: int = 17,
    device: str = "cpu",
) -> None:
    q = np.linspace(*q_range, grid_points)
    v = np.linspace(*v_range, grid_points)
    q_grid, v_grid = np.meshgrid(q, v)

    states = torch.tensor(
        np.stack((q_grid, v_grid), axis=-1),
        dtype=torch.float32,
    )

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

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].quiver(
        q_grid,
        v_grid,
        true_np[..., 0],
        true_np[..., 1],
    )
    axes[0].set_title("True vector field")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("v")

    axes[1].quiver(
        q_grid,
        v_grid,
        learned_np[..., 0],
        learned_np[..., 1],
    )
    axes[1].set_title("Learned vector field")
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("v")

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

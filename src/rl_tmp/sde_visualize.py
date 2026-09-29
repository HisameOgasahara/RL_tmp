import matplotlib.pyplot as plt
import torch


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


def plot_sample_paths(
    times: torch.Tensor,
    trajectories: torch.Tensor,
    title: str = "SDE sample paths",
) -> None:
    times_np = times.detach().cpu().numpy()
    values = trajectories.detach().cpu().numpy()

    if values.ndim == 2:
        values = values[:, None, :]

    plt.figure(figsize=(8, 4))
    for index in range(min(values.shape[1], 12)):
        plt.plot(times_np, values[:, index, 0], alpha=0.7)

    plt.xlabel("t")
    plt.ylabel("q(t)")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.show()


def plot_phase_samples(
    trajectories: torch.Tensor,
    title: str = "SDE phase-space sample paths",
) -> None:
    values = trajectories.detach().cpu().numpy()

    if values.ndim == 2:
        values = values[:, None, :]

    plt.figure(figsize=(7, 6))
    for index in range(min(values.shape[1], 12)):
        plt.plot(
            values[:, index, 0],
            values[:, index, 1],
            alpha=0.7,
        )

    plt.xlabel("q")
    plt.ylabel("v")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.show()


def plot_density(
    model: torch.nn.Module,
    time: float,
    q_range: tuple[float, float] = (-3.0, 3.0),
    v_range: tuple[float, float] = (-3.0, 3.0),
    grid_points: int = 100,
    device: str = "cpu",
    title: str = "Fokker-Planck PINN density",
) -> None:
    q = torch.linspace(*q_range, grid_points)
    v = torch.linspace(*v_range, grid_points)
    q_grid, v_grid = torch.meshgrid(q, v, indexing="xy")

    states = torch.stack(
        (q_grid.reshape(-1), v_grid.reshape(-1)),
        dim=-1,
    ).to(device)
    times = torch.full(
        (states.shape[0], 1),
        time,
        device=device,
    )

    with torch.no_grad():
        density = model(times, states).reshape(grid_points, grid_points).cpu()

    plt.figure(figsize=(7, 6))
    plt.contourf(
        q_grid.numpy(),
        v_grid.numpy(),
        density.numpy(),
        levels=30,
    )
    plt.xlabel("q")
    plt.ylabel("v")
    plt.title(title)
    plt.colorbar(label="density")
    plt.show()


def plot_neural_sde_comparison(
    reference: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    reference_np = reference.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.plot(
        reference_np[:, 0],
        reference_np[:, 1],
        label="reference SDE path",
        linewidth=2,
    )
    plt.plot(
        prediction_np[:, 0],
        prediction_np[:, 1],
        "--",
        label="Neural SDE path",
        linewidth=2,
    )
    plt.xlabel("q")
    plt.ylabel("v")
    plt.title("Reference vs learned SDE under the same noise")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def plot_sde_rl_rollout(
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
        label="uncontrolled SDE",
    )
    plt.plot(
        controlled_np[:, 0],
        controlled_np[:, 1],
        label="RL controlled SDE",
    )
    plt.scatter(goal_np[0], goal_np[1], marker="*", s=140, label="goal")
    plt.xlabel("q")
    plt.ylabel("v")
    plt.title("Stochastic RL control in phase space")
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

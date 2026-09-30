import matplotlib.pyplot as plt
import numpy as np
import torch

from .cartpole_system import (
    cartpole_field_numpy,
    cartpole_field_torch,
)


def _theta_phase_grid(
    theta_range=(-0.35, 0.35),
    theta_dot_range=(-1.0, 1.0),
    grid_points=21,
):
    theta = np.linspace(*theta_range, grid_points)
    theta_dot = np.linspace(*theta_dot_range, grid_points)

    theta_grid, theta_dot_grid = np.meshgrid(
        theta,
        theta_dot,
    )

    states = np.zeros(
        (
            grid_points,
            grid_points,
            4,
        ),
        dtype=np.float64,
    )

    states[..., 1] = theta_grid
    states[..., 3] = theta_dot_grid

    return (
        theta_grid,
        theta_dot_grid,
        states,
    )


def _project_theta_field(
    states,
    forces,
    config,
):
    flat_states = states.reshape(-1, 4)
    flat_forces = np.asarray(forces).reshape(-1)

    derivatives = np.stack(
        [
            cartpole_field_numpy(
                state,
                float(force),
                config,
            )
            for state, force in zip(
                flat_states,
                flat_forces,
            )
        ],
        axis=0,
    )

    derivatives = derivatives.reshape(
        states.shape,
    )

    return (
        derivatives[..., 1],
        derivatives[..., 3],
    )


def plot_state_trajectories(
    times,
    reference,
    prediction=None,
    labels=("DOP853", "prediction"),
    title="Cart-Pole trajectory",
):
    names = [
        "x",
        "theta",
        "x_dot",
        "theta_dot",
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
    )

    for index, axis in enumerate(
        axes.ravel()
    ):
        axis.plot(
            times[:len(reference)],
            reference[:, index],
            label=labels[0],
        )

        if prediction is not None:
            axis.plot(
                times[:len(prediction)],
                prediction[:, index],
                "--",
                label=labels[1],
            )

        axis.set_xlabel("t")
        axis.set_ylabel(names[index])
        axis.grid(alpha=0.2)
        axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    plt.show()


def plot_training(
    history,
    title,
):
    plt.figure(figsize=(8, 4))

    for key, values in history.items():
        plt.plot(
            values,
            label=key,
        )

    plt.xlabel("step/update")
    plt.ylabel("metric")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()


def plot_cartpole_solver_fields(
    config,
    uncontrolled,
    lqr_trajectory,
    lqr_gain,
    theta_range=(-0.35, 0.35),
    theta_dot_range=(-1.0, 1.0),
    grid_points=21,
):
    (
        theta_grid,
        theta_dot_grid,
        states,
    ) = _theta_phase_grid(
        theta_range=theta_range,
        theta_dot_range=theta_dot_range,
        grid_points=grid_points,
    )

    uncontrolled_force = np.zeros(
        (
            grid_points,
            grid_points,
        ),
        dtype=np.float64,
    )

    uncontrolled_dtheta, uncontrolled_dtheta_dot = (
        _project_theta_field(
            states=states,
            forces=uncontrolled_force,
            config=config,
        )
    )

    flat_states = states.reshape(-1, 4)

    lqr_force = -(
        flat_states
        @ np.asarray(lqr_gain).reshape(4, 1)
    ).reshape(
        grid_points,
        grid_points,
    )

    lqr_force = np.clip(
        lqr_force,
        -config.max_force,
        config.max_force,
    )

    lqr_dtheta, lqr_dtheta_dot = (
        _project_theta_field(
            states=states,
            forces=lqr_force,
            config=config,
        )
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    uncontrolled = np.asarray(uncontrolled)
    lqr_trajectory = np.asarray(lqr_trajectory)

    axes[0].quiver(
        theta_grid,
        theta_dot_grid,
        uncontrolled_dtheta,
        uncontrolled_dtheta_dot,
        alpha=0.7,
    )

    axes[0].plot(
        uncontrolled[:, 1],
        uncontrolled[:, 3],
        linewidth=2,
        label="uncontrolled trajectory",
    )

    axes[0].scatter(
        0.0,
        0.0,
        marker="*",
        s=140,
        label="goal",
    )

    axes[0].set_title(
        "Uncontrolled Cart-Pole field"
    )

    axes[1].quiver(
        theta_grid,
        theta_dot_grid,
        lqr_dtheta,
        lqr_dtheta_dot,
        alpha=0.7,
    )

    axes[1].plot(
        lqr_trajectory[:, 1],
        lqr_trajectory[:, 3],
        linewidth=2,
        label="LQR trajectory",
    )

    axes[1].scatter(
        0.0,
        0.0,
        marker="*",
        s=140,
        label="goal",
    )

    axes[1].set_title(
        "LQR closed-loop field"
    )

    for axis in axes:
        axis.set_xlabel("theta")
        axis.set_ylabel("theta_dot")
        axis.grid(alpha=0.2)
        axis.legend()

    figure.tight_layout()
    plt.show()


def plot_cartpole_pinn_phase_field(
    config,
    lqr_gain,
    reference,
    pinn_trajectory,
    theta_range=(-0.35, 0.35),
    theta_dot_range=(-1.0, 1.0),
    grid_points=21,
):
    (
        theta_grid,
        theta_dot_grid,
        states,
    ) = _theta_phase_grid(
        theta_range=theta_range,
        theta_dot_range=theta_dot_range,
        grid_points=grid_points,
    )

    flat_states = states.reshape(-1, 4)

    lqr_force = -(
        flat_states
        @ np.asarray(lqr_gain).reshape(4, 1)
    ).reshape(
        grid_points,
        grid_points,
    )

    lqr_force = np.clip(
        lqr_force,
        -config.max_force,
        config.max_force,
    )

    dtheta, dtheta_dot = (
        _project_theta_field(
            states=states,
            forces=lqr_force,
            config=config,
        )
    )

    reference = np.asarray(reference)
    pinn_trajectory = np.asarray(
        pinn_trajectory
    )

    plt.figure(figsize=(7, 6))

    plt.quiver(
        theta_grid,
        theta_dot_grid,
        dtheta,
        dtheta_dot,
        alpha=0.65,
    )

    plt.plot(
        reference[:, 1],
        reference[:, 3],
        linewidth=2,
        label="LQR + DOP853",
    )

    plt.plot(
        pinn_trajectory[:, 1],
        pinn_trajectory[:, 3],
        "--",
        linewidth=2,
        label="PINN trajectory",
    )

    plt.scatter(
        0.0,
        0.0,
        marker="*",
        s=140,
        label="goal",
    )

    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.title(
        "LQR closed-loop field + PINN trajectory"
    )
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()


def plot_cartpole_neural_ode_fields(
    config,
    learned_model,
    reference,
    learned_trajectory,
    theta_range=(-0.35, 0.35),
    theta_dot_range=(-1.0, 1.0),
    grid_points=21,
    device="cpu",
):
    (
        theta_grid,
        theta_dot_grid,
        states,
    ) = _theta_phase_grid(
        theta_range=theta_range,
        theta_dot_range=theta_dot_range,
        grid_points=grid_points,
    )

    zero_force = np.zeros(
        (
            grid_points,
            grid_points,
        ),
        dtype=np.float64,
    )

    true_dtheta, true_dtheta_dot = (
        _project_theta_field(
            states=states,
            forces=zero_force,
            config=config,
        )
    )

    states_tensor = torch.tensor(
        states.reshape(-1, 4),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        learned_field = learned_model(
            states_tensor
        ).cpu().numpy()

    learned_field = learned_field.reshape(
        grid_points,
        grid_points,
        4,
    )

    reference = np.asarray(reference)
    learned_trajectory = np.asarray(
        learned_trajectory
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].quiver(
        theta_grid,
        theta_dot_grid,
        true_dtheta,
        true_dtheta_dot,
        alpha=0.7,
    )

    axes[0].plot(
        reference[:, 1],
        reference[:, 3],
        linewidth=2,
        label="DOP853 reference",
    )

    axes[0].set_title(
        "True Cart-Pole vector field"
    )

    axes[1].quiver(
        theta_grid,
        theta_dot_grid,
        learned_field[..., 1],
        learned_field[..., 3],
        alpha=0.7,
    )

    axes[1].plot(
        learned_trajectory[:, 1],
        learned_trajectory[:, 3],
        linewidth=2,
        label="Neural ODE trajectory",
    )

    axes[1].set_title(
        "Learned Neural ODE vector field"
    )

    for axis in axes:
        axis.scatter(
            0.0,
            0.0,
            marker="*",
            s=140,
            label="goal",
        )
        axis.set_xlabel("theta")
        axis.set_ylabel("theta_dot")
        axis.grid(alpha=0.2)
        axis.legend()

    figure.tight_layout()
    plt.show()


def plot_cartpole_rl_closed_loop_fields(
    policy,
    config,
    lqr_gain,
    lqr_trajectory,
    rl_trajectory,
    goal_state,
    theta_range=(-0.35, 0.35),
    theta_dot_range=(-1.0, 1.0),
    grid_points=21,
    device="cpu",
):
    (
        theta_grid,
        theta_dot_grid,
        states,
    ) = _theta_phase_grid(
        theta_range=theta_range,
        theta_dot_range=theta_dot_range,
        grid_points=grid_points,
    )

    flat_states = states.reshape(-1, 4)

    lqr_force = -(
        flat_states
        @ np.asarray(lqr_gain).reshape(4, 1)
    ).reshape(
        grid_points,
        grid_points,
    )

    lqr_force = np.clip(
        lqr_force,
        -config.max_force,
        config.max_force,
    )

    lqr_dtheta, lqr_dtheta_dot = (
        _project_theta_field(
            states=states,
            forces=lqr_force,
            config=config,
        )
    )

    flat_states_tensor = torch.tensor(
        flat_states,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        rl_force = policy(
            flat_states_tensor,
            max_force=config.max_force,
        ).cpu().numpy()

    rl_force = rl_force.reshape(
        grid_points,
        grid_points,
    )

    rl_dtheta, rl_dtheta_dot = (
        _project_theta_field(
            states=states,
            forces=rl_force,
            config=config,
        )
    )

    lqr_trajectory = np.asarray(
        lqr_trajectory
    )

    if torch.is_tensor(rl_trajectory):
        rl_trajectory = (
            rl_trajectory
            .detach()
            .cpu()
            .numpy()
        )
    else:
        rl_trajectory = np.asarray(
            rl_trajectory
        )

    goal_state = np.asarray(goal_state)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].quiver(
        theta_grid,
        theta_dot_grid,
        lqr_dtheta,
        lqr_dtheta_dot,
        alpha=0.7,
    )

    axes[0].plot(
        lqr_trajectory[:, 1],
        lqr_trajectory[:, 3],
        linewidth=2,
        label="LQR trajectory",
    )

    axes[0].set_title(
        "LQR closed-loop vector field"
    )

    axes[1].quiver(
        theta_grid,
        theta_dot_grid,
        rl_dtheta,
        rl_dtheta_dot,
        alpha=0.7,
    )

    axes[1].plot(
        rl_trajectory[:, 1],
        rl_trajectory[:, 3],
        linewidth=2,
        label="RL trajectory",
    )

    axes[1].set_title(
        "RL closed-loop vector field"
    )

    for axis in axes:
        axis.scatter(
            goal_state[1],
            goal_state[3],
            marker="*",
            s=140,
            label="goal",
        )
        axis.set_xlabel("theta")
        axis.set_ylabel("theta_dot")
        axis.grid(alpha=0.2)
        axis.legend()

    figure.tight_layout()
    plt.show()

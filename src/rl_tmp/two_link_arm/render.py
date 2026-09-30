from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from .dynamics import (
    TwoLinkConfig,
    forward_kinematics,
)


def rollout_policy(
    policy,
    env,
    seed: int = 0,
):
    observation, info = env.reset(
        seed=seed
    )

    states = [
        info["state"].copy()
    ]

    actions = []
    rewards = []
    end_effectors = [
        forward_kinematics(
            info["state"][:2],
            env.config,
        )
    ]

    while True:
        action = np.asarray(
            policy(observation),
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        states.append(
            info["state"].copy()
        )

        actions.append(
            action.copy()
        )

        rewards.append(
            float(reward)
        )

        end_effectors.append(
            info[
                "end_effector"
            ].copy()
        )

        if terminated or truncated:
            break

    return {
        "states": np.asarray(
            states,
            dtype=np.float64,
        ),
        "actions": np.asarray(
            actions,
            dtype=np.float64,
        ),
        "rewards": np.asarray(
            rewards,
            dtype=np.float64,
        ),
        "end_effectors": np.asarray(
            end_effectors,
            dtype=np.float64,
        ),
        "target": env.target.copy(),
        "dt": env.dt,
    }


def stable_baselines_policy(
    model,
    deterministic: bool = True,
):
    def policy(observation):
        action, _ = model.predict(
            observation,
            deterministic=deterministic,
        )

        return action

    return policy


def _make_box(
    client,
    length,
    half_width,
):
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.5 * length,
            half_width,
            half_width,
        ],
        rgbaColor=[
            0.25,
            0.55,
            0.95,
            1.0,
        ],
        physicsClientId=client,
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=[
            0.0,
            0.0,
            0.0,
        ],
        physicsClientId=client,
    )


def _make_sphere(
    client,
    radius,
    color,
):
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client,
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=[
            0.0,
            0.0,
            0.0,
        ],
        physicsClientId=client,
    )


def _set_link_pose(
    body,
    x0,
    z0,
    x1,
    z1,
    client,
):
    cx = 0.5 * (x0 + x1)
    cz = 0.5 * (z0 + z1)

    angle = np.arctan2(
        z1 - z0,
        x1 - x0,
    )

    quaternion = p.getQuaternionFromEuler(
        [
            0.0,
            -angle,
            0.0,
        ]
    )

    p.resetBasePositionAndOrientation(
        body,
        [
            cx,
            0.0,
            cz,
        ],
        quaternion,
        physicsClientId=client,
    )


def _pybullet_frames(
    states,
    target,
    config: TwoLinkConfig,
    width: int,
    height: int,
):
    client = p.connect(
        p.DIRECT
    )

    p.resetSimulation(
        physicsClientId=client
    )

    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI,
        0,
        physicsClientId=client,
    )

    link1_body = _make_box(
        client,
        config.link1,
        0.065,
    )

    link2_body = _make_box(
        client,
        config.link2,
        0.055,
    )

    base_body = _make_sphere(
        client,
        0.10,
        [
            0.2,
            0.2,
            0.2,
            1.0,
        ],
    )

    joint_body = _make_sphere(
        client,
        0.09,
        [
            0.95,
            0.55,
            0.15,
            1.0,
        ],
    )

    end_body = _make_sphere(
        client,
        0.085,
        [
            0.2,
            0.85,
            0.35,
            1.0,
        ],
    )

    target_body = _make_sphere(
        client,
        0.10,
        [
            0.95,
            0.25,
            0.25,
            1.0,
        ],
    )

    p.resetBasePositionAndOrientation(
        base_body,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        physicsClientId=client,
    )

    p.resetBasePositionAndOrientation(
        target_body,
        [
            float(target[0]),
            0.0,
            float(target[1]),
        ],
        [0.0, 0.0, 0.0, 1.0],
        physicsClientId=client,
    )

    view = p.computeViewMatrix(
        cameraEyePosition=[
            0.0,
            -5.0,
            1.0,
        ],
        cameraTargetPosition=[
            0.0,
            0.0,
            0.5,
        ],
        cameraUpVector=[
            0.0,
            0.0,
            1.0,
        ],
    )

    projection = p.computeProjectionMatrixFOV(
        fov=48.0,
        aspect=width / height,
        nearVal=0.05,
        farVal=20.0,
    )

    frames = []

    for state in states:
        q1, q2 = state[:2]

        x1 = (
            config.link1
            * np.cos(q1)
        )

        z1 = (
            config.link1
            * np.sin(q1)
        )

        x2 = (
            x1
            + config.link2
            * np.cos(
                q1 + q2
            )
        )

        z2 = (
            z1
            + config.link2
            * np.sin(
                q1 + q2
            )
        )

        _set_link_pose(
            link1_body,
            0.0,
            0.0,
            x1,
            z1,
            client,
        )

        _set_link_pose(
            link2_body,
            x1,
            z1,
            x2,
            z2,
            client,
        )

        p.resetBasePositionAndOrientation(
            joint_body,
            [
                float(x1),
                0.0,
                float(z1),
            ],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client,
        )

        p.resetBasePositionAndOrientation(
            end_body,
            [
                float(x2),
                0.0,
                float(z2),
            ],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client,
        )

        image = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=client,
        )

        rgba = np.asarray(
            image[2],
            dtype=np.uint8,
        ).reshape(
            height,
            width,
            4,
        )

        frames.append(
            rgba[:, :, :3]
        )

    p.disconnect(
        physicsClientId=client
    )

    return frames


def render_rollout_video(
    rollout,
    output_path,
    config: TwoLinkConfig | None = None,
    fps: int = 25,
    frame_stride: int = 2,
):
    if config is None:
        config = TwoLinkConfig()

    states = rollout[
        "states"
    ]

    target = rollout[
        "target"
    ]

    bullet_frames = _pybullet_frames(
        states=states,
        target=target,
        config=config,
        width=520,
        height=420,
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_frames = []

    q1 = states[:, 0]
    q2 = states[:, 1]
    dq1 = states[:, 2]
    dq2 = states[:, 3]

    for index in range(
        0,
        len(states),
        max(
            1,
            int(frame_stride),
        ),
    ):
        figure = plt.figure(
            figsize=(
                12,
                6,
            )
        )

        grid = figure.add_gridspec(
            2,
            3,
            width_ratios=[
                1.45,
                1.0,
                1.0,
            ],
        )

        ax_robot = figure.add_subplot(
            grid[:, 0]
        )

        ax_q = figure.add_subplot(
            grid[0, 1]
        )

        ax_p1 = figure.add_subplot(
            grid[0, 2]
        )

        ax_p2 = figure.add_subplot(
            grid[1, 1]
        )

        ax_time = figure.add_subplot(
            grid[1, 2]
        )

        ax_robot.imshow(
            bullet_frames[index]
        )

        ax_robot.set_title(
            "PyBullet 2-link arm"
        )

        ax_robot.axis(
            "off"
        )

        ax_q.plot(
            q1[: index + 1],
            q2[: index + 1],
        )

        ax_q.scatter(
            q1[index],
            q2[index],
            s=35,
        )

        ax_q.set_xlabel(
            r"$q_1$"
        )

        ax_q.set_ylabel(
            r"$q_2$"
        )

        ax_q.set_title(
            r"configuration space $(q_1,q_2)$"
        )

        ax_p1.plot(
            q1[: index + 1],
            dq1[: index + 1],
        )

        ax_p1.scatter(
            q1[index],
            dq1[index],
            s=35,
        )

        ax_p1.set_xlabel(
            r"$q_1$"
        )

        ax_p1.set_ylabel(
            r"$\dot q_1$"
        )

        ax_p1.set_title(
            r"phase space $(q_1,\dot q_1)$"
        )

        ax_p2.plot(
            q2[: index + 1],
            dq2[: index + 1],
        )

        ax_p2.scatter(
            q2[index],
            dq2[index],
            s=35,
        )

        ax_p2.set_xlabel(
            r"$q_2$"
        )

        ax_p2.set_ylabel(
            r"$\dot q_2$"
        )

        ax_p2.set_title(
            r"phase space $(q_2,\dot q_2)$"
        )

        time = (
            np.arange(
                index + 1
            )
            * rollout[
                "dt"
            ]
        )

        ax_time.plot(
            time,
            q1[: index + 1],
            label=r"$q_1$",
        )

        ax_time.plot(
            time,
            q2[: index + 1],
            label=r"$q_2$",
        )

        ax_time.set_xlabel(
            "time"
        )

        ax_time.set_ylabel(
            "joint angle"
        )

        ax_time.set_title(
            "Euler-Lagrange trajectory"
        )

        ax_time.legend()

        figure.tight_layout()

        figure.canvas.draw()

        rgba = np.asarray(
            figure.canvas.buffer_rgba()
        )

        video_frames.append(
            rgba[:, :, :3].copy()
        )

        plt.close(
            figure
        )

    imageio.mimsave(
        output_path,
        video_frames,
        fps=fps,
        codec="libx264",
        quality=7,
    )

    return output_path

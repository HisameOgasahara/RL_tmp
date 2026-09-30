from dataclasses import dataclass

import numpy as np


@dataclass
class TwoLinkConfig:
    link1: float = 1.0
    link2: float = 1.0
    mass1: float = 1.0
    mass2: float = 1.0
    inertia1: float = 1.0 / 12.0
    inertia2: float = 1.0 / 12.0
    gravity: float = 9.81
    damping1: float = 0.08
    damping2: float = 0.08
    max_torque: float = 8.0


def forward_kinematics(q, config: TwoLinkConfig):
    q = np.asarray(q, dtype=np.float64)
    q1 = q[..., 0]
    q2 = q[..., 1]

    x1 = config.link1 * np.cos(q1)
    z1 = config.link1 * np.sin(q1)

    x2 = x1 + config.link2 * np.cos(q1 + q2)
    z2 = z1 + config.link2 * np.sin(q1 + q2)

    return np.stack([x2, z2], axis=-1)


def mass_matrix(q, config: TwoLinkConfig):
    q1, q2 = np.asarray(q, dtype=np.float64)

    l1 = config.link1
    lc1 = 0.5 * config.link1
    lc2 = 0.5 * config.link2
    m1 = config.mass1
    m2 = config.mass2
    i1 = config.inertia1
    i2 = config.inertia2

    c2 = np.cos(q2)

    m11 = (
        i1
        + i2
        + m1 * lc1 * lc1
        + m2 * (
            l1 * l1
            + lc2 * lc2
            + 2.0 * l1 * lc2 * c2
        )
    )

    m12 = (
        i2
        + m2 * (
            lc2 * lc2
            + l1 * lc2 * c2
        )
    )

    m22 = i2 + m2 * lc2 * lc2

    return np.array(
        [
            [m11, m12],
            [m12, m22],
        ],
        dtype=np.float64,
    )


def coriolis_vector(q, dq, config: TwoLinkConfig):
    _, q2 = np.asarray(q, dtype=np.float64)
    dq1, dq2 = np.asarray(dq, dtype=np.float64)

    h = (
        config.mass2
        * config.link1
        * (0.5 * config.link2)
        * np.sin(q2)
    )

    return np.array(
        [
            -h * (2.0 * dq1 * dq2 + dq2 * dq2),
            h * dq1 * dq1,
        ],
        dtype=np.float64,
    )


def gravity_vector(q, config: TwoLinkConfig):
    q1, q2 = np.asarray(q, dtype=np.float64)

    l1 = config.link1
    lc1 = 0.5 * config.link1
    lc2 = 0.5 * config.link2
    m1 = config.mass1
    m2 = config.mass2
    g = config.gravity

    g1 = (
        (m1 * lc1 + m2 * l1)
        * g
        * np.cos(q1)
        + m2
        * lc2
        * g
        * np.cos(q1 + q2)
    )

    g2 = (
        m2
        * lc2
        * g
        * np.cos(q1 + q2)
    )

    return np.array(
        [g1, g2],
        dtype=np.float64,
    )


def dynamics(state, torque, config: TwoLinkConfig):
    state = np.asarray(state, dtype=np.float64)
    torque = np.asarray(torque, dtype=np.float64)

    q = state[:2]
    dq = state[2:]

    torque = np.clip(
        torque,
        -config.max_torque,
        config.max_torque,
    )

    damping = np.array(
        [
            config.damping1 * dq[0],
            config.damping2 * dq[1],
        ],
        dtype=np.float64,
    )

    rhs = (
        torque
        - coriolis_vector(q, dq, config)
        - gravity_vector(q, config)
        - damping
    )

    ddq = np.linalg.solve(
        mass_matrix(q, config),
        rhs,
    )

    return np.concatenate(
        [
            dq,
            ddq,
        ]
    )


def rk4_step(state, torque, dt: float, config: TwoLinkConfig):
    state = np.asarray(state, dtype=np.float64)

    k1 = dynamics(
        state,
        torque,
        config,
    )

    k2 = dynamics(
        state + 0.5 * dt * k1,
        torque,
        config,
    )

    k3 = dynamics(
        state + 0.5 * dt * k2,
        torque,
        config,
    )

    k4 = dynamics(
        state + dt * k3,
        torque,
        config,
    )

    next_state = state + (
        dt
        / 6.0
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )

    next_state[:2] = (
        next_state[:2] + np.pi
    ) % (2.0 * np.pi) - np.pi

    return next_state

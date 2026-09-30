# PyBullet 2-link arm + Euler-Lagrange state-space visualization

## 목적

같은 2-link arm reaching 문제에서 CEM, PPO, SAC를 각각 독립적으로 학습하고 비교합니다.

세 알고리즘은 하나의 Colab 노트북 안에서 섹션을 나눠 각각 독립적으로 학습합니다.

- `notebooks/two_link_arm_colab.ipynb`

공통 환경과 Euler-Lagrange dynamics는 `src/rl_tmp/two_link_arm/`에 있습니다.

## 상태와 제어입력

관절각을

$$
q=
\begin{pmatrix}
q_1\\
q_2
\end{pmatrix}
\in\mathbb{R}^2
$$

라고 하고 관절각속도를

$$
\dot q=
\begin{pmatrix}
\dot q_1\\
\dot q_2
\end{pmatrix}
\in\mathbb{R}^2
$$

라고 합니다.

상태는

$$
x=
\begin{pmatrix}
q_1\\
q_2\\
\dot q_1\\
\dot q_2
\end{pmatrix}
\in\mathbb{R}^4
$$

이고 RL 정책의 제어입력은 두 관절에 가하는 torque

$$
u=
\tau=
\begin{pmatrix}
\tau_1\\
\tau_2
\end{pmatrix}
\in\mathbb{R}^2
$$

입니다.

## Euler-Lagrange equation

시뮬레이션에 사용하는 기계계는

$$
M(q)\ddot q
+
C(q,\dot q)\dot q
+
g(q)
+
B\dot q
=
\tau
$$

로 둡니다.

여기서

- $M(q)\in\mathbb{R}^{2\times2}$: mass matrix
- $C(q,\dot q)\dot q\in\mathbb{R}^2$: Coriolis/centrifugal term
- $g(q)\in\mathbb{R}^2$: gravity term
- $B\in\mathbb{R}^{2\times2}$: joint damping matrix
- $\tau\in\mathbb{R}^2$: RL policy가 출력한 torque

입니다.

이를 1차 상태방정식

$$
\dot x=f(x,u),
\qquad
f:\mathbb{R}^4\times\mathbb{R}^2\to\mathbb{R}^4
$$

로 바꿔 RK4로 적분합니다.

PyBullet은 Colab에서 GUI를 띄우는 대신 DIRECT mode renderer로 사용합니다. 즉 실제 운동 상태는 위 Euler-Lagrange 방정식에서 만들고, PyBullet에는 동일한 $q_1,q_2$를 넣어 같은 시각의 로봇팔 자세를 렌더링합니다.

그래서 영상의 로봇팔 동작과 오른쪽 상태공간 trajectory는 같은 rollout입니다.

## 동기화 시각화

각 영상 frame에서 동시에 다음을 표시합니다.

$$
(q_1,q_2)
$$

configuration-space trajectory,

$$
(q_1,\dot q_1)
$$

첫 번째 관절 phase-space trajectory,

$$
(q_2,\dot q_2)
$$

두 번째 관절 phase-space trajectory와 시간에 따른 $q_1,q_2$ 변화입니다.

## 알고리즘 비교

### CEM

정책 parameter vector 자체를 분포에서 샘플링하고 rollout reward가 높은 elite 집합으로 sampling distribution을 갱신합니다.

신경망 gradient를 쓰지 않는 policy search라 PPO/SAC와 탐색 방식이 가장 다르게 보입니다.

### PPO

현재 policy로 rollout을 생성하는 on-policy actor-critic 방법입니다.

노트북에서는 early / middle / final policy를 각각 꺼내 같은 2-link arm 환경에서 실행해 학습 진행에 따라 궤적이 어떻게 변하는지 봅니다.

### SAC

replay buffer를 사용하는 off-policy actor-critic 방법이며 entropy objective를 포함합니다.

PPO와 같은 연속 torque action space에서 학습하므로, 같은 물리계에서 on-policy와 off-policy 탐색의 차이를 비교할 수 있습니다.

## 코드

- Euler-Lagrange dynamics: `src/rl_tmp/two_link_arm/dynamics.py`
- Gymnasium environment: `src/rl_tmp/two_link_arm/env.py`
- CEM: `src/rl_tmp/two_link_arm/cem.py`
- PPO: `src/rl_tmp/two_link_arm/ppo.py`
- SAC: `src/rl_tmp/two_link_arm/sac.py`
- PyBullet + state-space synchronized video: `src/rl_tmp/two_link_arm/render.py`

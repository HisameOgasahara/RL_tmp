# ODE methods

## System

감쇠 조화진동자의 상태는

$$
x(t)=
\begin{pmatrix}
q(t)\\
v(t)
\end{pmatrix}
\in\mathbb{R}^2
$$

이고,

$$
\dot q(t)=v(t),
\qquad
\dot v(t)=-kq(t)-cv(t)
$$

를 사용합니다.

RL에서는 제어입력

$$
u(t)\in\mathbb{R}
$$

를 추가해

$$
\dot v(t)=-kq(t)-cv(t)+u(t)
$$

로 둡니다.

## Notebook

- Colab: `notebooks/ode_methods_colab.ipynb`
- Core code: `src/rl_tmp/ode_methods/`

노트북에서는 solver, Neural ODE, PINN, RL을 같은 계에서 비교하고 각 방법의 결과 궤적과 벡터장을 시각화합니다.

# RL_tmp

## ODE

[![Open ODE In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/ode_methods_colab.ipynb)

감쇠 조화진동자

$$
x(t)=(q(t),v(t))\in\mathbb{R}^2,
$$

$$
\dot q(t)=v(t),\qquad
\dot v(t)=-kq(t)-cv(t).
$$

RL에서는 제어입력 $u(t)\in\mathbb{R}$를 추가합니다.

$$
\dot v(t)=-kq(t)-cv(t)+u(t).
$$

Colab: `notebooks/ode_methods_colab.ipynb`

## SDE

[![Open SDE In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/sde_methods_colab.ipynb)

같은 계에 velocity noise를 추가합니다. $W_t$는 1차원 Wiener process입니다.

$$
dq_t=v_t\,dt,
$$

$$
dv_t=(-kq_t-cv_t)\,dt+\sigma\,dW_t.
$$

RL에서는

$$
dv_t=(-kq_t-cv_t+u_t)\,dt+\sigma\,dW_t
$$

를 사용합니다.

SDE sample path는 일반적으로 미분 가능하지 않으므로 PINN은 path가 아니라 Fokker–Planck equation을 대상으로 합니다.

Colab: `notebooks/sde_methods_colab.ipynb`

## Nonlinear Cart-Pole Control

[![Open Cart-Pole In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/cartpole_control_colab.ipynb)

상태와 제어입력은

```math
x(t)=
\begin{pmatrix}
p(t)\\
\theta(t)\\
\dot p(t)\\
\dot\theta(t)
\end{pmatrix}
\in\mathbb{R}^4,
\qquad
u(t)\in\mathbb{R}.
```

$m_c>0$는 cart mass, $m_p>0$는 pole mass, $\ell>0$는 pole half-length, $g>0$는 중력가속도이고

```math
M=m_c+m_p
```

```math
a(x,u)=\frac{u+m_p\ell\dot\theta^2\sin\theta}{M}
```

라 두면 현재 코드의 비선형 Cart-Pole dynamics는

```math
\dot p=\dot p,
\qquad
\dot\theta=\dot\theta
```

```math
\ddot\theta=
\frac{
g\sin\theta-\cos\theta\,a(x,u)
}{
\ell\left(
\frac{4}{3}
-\frac{m_p\cos^2\theta}{M}
\right)
}
```

```math
\ddot p=
a(x,u)
-
\frac{m_p\ell\ddot\theta\cos\theta}{M}
```

즉

```math
\dot x(t)=f(x(t),u(t)),
\qquad
f:\mathbb{R}^4\times\mathbb{R}\to\mathbb{R}^4
```

목표 상태는

```math
x_{\mathrm{goal}}=(0,0,0,0)
```

전체 vector field는 4차원이므로 한 평면에 직접 그릴 수 없습니다. 노트북에서는 $p=\dot p=0$으로 고정한 $(\theta,\dot\theta)$ 2차원 slice에서 uncontrolled / LQR / learned closed-loop vector field와 trajectory를 시각화합니다.

Colab: `notebooks/cartpole_control_colab.ipynb`

핵심 구현은 `src/rl_tmp/*.py`에 있습니다.

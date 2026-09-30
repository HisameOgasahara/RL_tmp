# SDE methods

## System

감쇠 조화진동자에 velocity noise를 추가합니다.

$$
dq_t=v_t\,dt
$$

$$
dv_t=(-kq_t-cv_t)\,dt+\sigma\,dW_t
$$

여기서

$$
W_t:\Omega\times\mathbb{R}_{\ge0}\to\mathbb{R}
$$

는 1차원 Wiener process입니다.

RL에서는 제어입력

$$
u_t\in\mathbb{R}
$$

를 추가해

$$
dv_t=(-kq_t-cv_t+u_t)\,dt+\sigma\,dW_t
$$

를 사용합니다.

SDE sample path는 일반적으로 미분 가능하지 않으므로 PINN은 개별 path가 아니라 Fokker–Planck equation을 대상으로 합니다.

## Notebook

- Colab: `notebooks/sde_methods_colab.ipynb`
- Core code: `src/rl_tmp/sde_methods/`

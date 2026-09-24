# CNRK2 time integration

CNRK2 has three stages and overall order two. For the signed explicit
acceleration ``N_j=\mathcal N+f``, its coefficients are

```math
A=(0,-5/9,-153/128),\quad
B=(1/3,15/16,8/15),\quad
C=(1/6,5/24,1/8).
```

At stage times ``t+(0,1/3,3/4)\Delta t``, update ``Q=A_jQ+N_j`` and solve

```math
(\lambda_j-\nu\Delta)u_{new}+\nabla q_{new}+G_{new}
=\lambda_j u+\nu\Delta u-\nabla q-G_{old}
 +(B_j/C_j)Q+2\nu U_b''e_x,
\qquad \lambda_j=1/(C_j\Delta t).
```

The first stage overwrites Q. Three banks of factors correspond to the three
shifts; no factorisation occurs during a nominal fixed step. The base-flow
curvature appears twice because it enters both Crank–Nicolson halves.

`stagepressure(state)` is the algebraic pressure retained by this discrete
scheme. It is needed in the next stage and must be saved for an exact restart.
It has no independent evolution equation. `pressure(U, problem, t)` instead
reconstructs instantaneous physical/modified pressure from a divergence-free
velocity, using

```math
\Delta q=\nabla\cdot(\mathcal N+f),\qquad
Dq|_{\pm1}=(\mathcal N+f)_y+\nu D^2v|_{\pm1}.
```

Viscosity drops out of the interior divergence but remains in the wall data.
Use reconstructed pressure to initialise a state; do not replace stage pressure
after every timestep.

`Flows.flow(problem)` uses fixed nominal steps and can shorten the terminal
step to hit the requested time. A shortened step builds temporary factors on
the active backend. For evenly spaced sampling, use an integer number of
nominal steps and carry the final time into the next run.

## Accuracy and interpretation

The name **CNRK2** follows Gibson's Channelflow: Crank–Nicolson treatment of
viscous terms coupled to an explicit Runge–Kutta history update. The three
substeps do not make this a third-order method. The coefficient arrays and
stage shifts correspond to [`RungeKuttaDNS`](https://github.com/epfl-ecps/channelflow/blob/master/channelflow/dnsalgo.cpp).

The stage pressure is part of the algebraic state of this scheme. Even in the
convective form it is not a promise of an independently time-accurate pressure
sample at every substep. Use `pressure(U, problem, t)` when instantaneous pressure
consistent with the current velocity is needed, and retain `stagepressure(state)`
for continuation and restart.

Choose the nominal timestep for the explicit advective stability restriction;
implicit diffusion does not remove that restriction. Compare results at smaller
timesteps independently of spatial refinement. No embedded error estimator or
automatic CFL adjustment is supplied.

# zflows — source tree

PyTorch normalizing flows for unconditional energy-based sampling
(strongly inspired by [`zuko`](https://github.com/probabilists/zuko)).

```
zflows
├── __init__.py          # package root — re-exports the public surface
├── flow.py              # public flows: Flow, NSF, NCSF, CNF, OTFlow, RealNVP
├── potential.py         # energy/target potentials and base distributions
├── loss.py              # training objectives (KL / OT) + compilation helpers
├── utils.py             # sampling, MCMC, importance sampling, diagnostics
└── core                 # internal machinery (no public/conditional plumbing)
    ├── __init__.py
    ├── transforms.py    # bijective Transform subclasses
    ├── flows.py         # lazy nn.Module wrappers → concrete Transforms
    ├── nn.py            # Linear / MLP / masked-MLP building blocks
    ├── numerics.py      # tensor & autograd utilities (RK4, bisection, quadrature)
    └── otflow.py        # OT-Flow potential network + transform
```

---

## `flow.py` — public normalizing flows
Abstract base whose subclasses implement `.t() -> ComposedTransform`.

- `class Flow(nn.Module, ABC)` — abstract flow base; `.t()`, `.zeros()`
- `class NSF(Flow)` — Neural Spline Flow on `[a, b]^d` (translation-sandwiched RQS)
- `class NCSF(Flow)` — Neural Circular Spline Flow (periodic per coordinate)
- `class CNF(Flow)` — Continuous Normalizing Flow (FFJORD-style ODE)
- `class OTFlow(Flow)` — Optimal-Transport continuous flow
- `class RealNVP(Flow)` — coupling-based affine flow

## `potential.py` — potentials & base distributions
- `class Potential(nn.Module)` — abstract energy base; `forward`, `grad`, `eval`, `enable_grad`, `enable_eval`
- `def potential_from(fn) -> Potential` — wrap a callable as a `Potential`
- `class Linear_Combination(Potential)` — weighted blend of potentials; `set_coeffs`, compiled grad/eval
- `def linear_combination(...)` — constructor helper for `Linear_Combination`
- `class Uniform(Potential)` — uniform on a box; `forward`, `samples`
- `class Gaussian(Potential)` — Gaussian potential; `forward`, `samples`
- `class Gaussian_Mixture(Potential)` — mixture of Gaussians; `forward`, `samples`

## `loss.py` — training objectives
- `def reverse_KL_F` / `def reverse_KL_G` — reverse-KL losses (forward / inverse map)
- `def forward_KL_F` / `def forward_KL_G` — forward-KL losses (forward / inverse map)
- `def OT_loss` — optimal-transport objective for `OTFlow`
- `def loss_compile` — `torch.compile` wrapper for a loss
- `def loss_compile_beta` — compiled loss parameterised by temperature `beta`

## `utils.py` — sampling, MCMC & diagnostics
- `def compute_ESS` / `compute_ESS_log` — effective sample size
- `def compute_CESS` / `compute_CESS_log` — conditional ESS
- `def importance_weights_F` / `_G` (+ `_log_` variants) — importance weights through a flow
- `def resample` — multinomial resampling
- `def lbfgs` — batched L-BFGS minimisation of a potential
- `def langevin` — (adjusted) Langevin dynamics
- `def sequential_monte_carlo` — SMC sampler over a temperature ladder
- `def annealed_importance_sampling_F` / `_G` — AIS through a flow
- `def stochastic_heun` — stochastic Heun SDE integrator
- `def hamiltonian_monte_carlo` — HMC sampler
- `def check_compile_available` — probe `torch.compile` support
- `def set_cache_size_limit` — tune dynamo cache size
- `def suppress_warnings` — silence noisy warnings

---

## `core/transforms.py` — bijective transforms
Subset of `zuko/transforms.py`; `MonotonicRQSTransform`/`CircularShiftTransform`
accept a per-coordinate `bound`.

- `class ComposedTransform(Transform)` — chain of transforms; `call_and_ladj`, `inv`, compiled `for_ladj`/`inv_ladj`
- `class DependentTransform(Transform)` — reinterpret batch dims as event dims
- `class IdentityTransform(Transform)`
- `class AdditiveTransform(Transform)` — constant shift
- `class MonotonicAffineTransform(Transform)`
- `class MonotonicRQSTransform(Transform)` — rational-quadratic spline; `bin`, `searchsorted`
- `class CircularShiftTransform(Transform)` — periodic shift
- `class AutoregressiveTransform(Transform)`
- `class CouplingTransform(Transform)` — `split` / `merge`
- `class FreeFormJacobianTransform(Transform)` — continuous-flow ODE transform
- `class RotationTransform(Transform)`
- `class LULinearTransform(Transform)` — LU-parameterised linear map

## `core/flows.py` — lazy transform modules
`nn.Module`s whose `forward()` (no args) returns a concrete `Transform`.

- `class MaskedAutoregressiveTransform(nn.Module)` — MAF conditioner
- `class GeneralCouplingTransform(nn.Module)`
- `class FFJTransform(nn.Module)` — free-form Jacobian / FFJORD
- `class OTFlowLazy(nn.Module)`
- `def CircularRQSTransform(...)` — circular RQS builder
- `class LinearMixingTransform(nn.Module)` — rotation / LU mixing layer

## `core/nn.py` — neural-net building blocks
- `def linear(x, W, b)` — functional linear op
- `class Linear(nn.Module)`
- `class MLP(nn.Sequential)`
- `class MaskedLinear(nn.Linear)`
- `class MaskedMLP(nn.Sequential)` — autoregressive masked MLP

## `core/numerics.py` — tensor & autograd utilities
- `class Partial(nn.Module)` — `nn.Module` wrapper of `functools.partial`
- `def bisection` / `class Bisection(autograd.Function)` — implicit-grad root finder
- `def broadcast` — broadcast over leading dims
- `def gauss_legendre` / `class GaussLegendre(autograd.Function)` — quadrature (`nodes`, `quadrature`)
- `def rk4_fixed` — fixed-step RK4 ODE integrator
- `def unpack` — split a flat tensor into shaped pieces

## `core/otflow.py` — OT-Flow machinery
Ported from Onken et al., "OT-Flow" (AAAI 2021).

- `def antideriv_tanh` / `def deriv_tanh` — tanh helpers
- `class ResNN(nn.Module)` — residual network
- `class OTPhi(nn.Module)` — potential network; `trHess` (trace of Hessian)
- `class OTFlowTransform(Transform)` — `_drift`, `call_and_ladj`, `call_full`, `log_abs_det_jacobian`

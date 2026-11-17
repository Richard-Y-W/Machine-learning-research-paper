# Cycle 4: internal features versus prediction-only residual adaptation

Written before execution. This cycle changes the task explicitly: adapters now
receive a few noisy acceleration measurements from the changed environment.
They receive no physical mechanism library, coefficient values, or intervention
formula. This is few-shot dynamics adaptation, not specification-only editing.

## Scope and data

Source systems: Duffing, pendulum, a spring with contact forces, and a van der Pol
type oscillator with state-dependent damping. All have observed state (q,v),
action u, and acceleration output. Source networks remain 3-32-32-1 ReLU MLPs,
trained on 4,096 noiseless source labels in [-1.5,1.5]^3 for 2,400 Adam steps.

Two changed environments per source:
- Parameter shift: first restoring coefficient x1.5 and viscous damping x2.
- Added mechanisms: extra restoring force -0.2 sin(3q) and friction -0.3 tanh(4v).

Adapters are not told which shift occurred. They receive 16 or 64 broad uniform
changed-environment acceleration observations with independent Gaussian noise
standard deviation 0.01. All methods share the same observations and noiseless
evaluation points. Source-model predictions are callable during adaptation and
deployment. Those calls do not consume changed-environment labels, but are
counted as a distinct access assumption. Arbitrary state resets and direct
acceleration measurements are simulator conveniences, not real-world guarantees.

## Methods and access contracts

- Source only.
- Warm full-network L-BFGS: copies all source weights.
- Fresh full-network L-BFGS: same architecture, independent initialization.
- Frozen learned features: calibrate an additive output-head correction and fold
  it into the original model. Requires hidden activation access.
- Prediction-only linear residual: source prediction plus generic affine fit.
- Prediction-only cubic residual: all monomials through total degree three, not
  a named physical mechanism library.
- Prediction-only RBF residual: kernel ridge on source-prediction errors.
- Prediction-only ARD RBF residual: the same kernel method with independent
  coordinate lengths selected by calibration leave-one-out error.
- Prediction-only neural residual: a fresh MLP learns a correction around the
  callable source predictor; its output layer starts at zero.

The residual methods reuse source behavior without reading pretrained weights.
They deploy a wrapper and thus differ in architecture/storage from in-place
weight edits. Record these differences; do not claim equal-capacity superiority.
Pretrained-feature advantage must be assessed against these functional-reuse
baselines, not just a fresh full-network student.

## Selection rules fixed before fresh evaluation

Ridge choices: 1e-5,1e-4,1e-3,1e-2,0.1, applied as n*lambda in the Gram solve.
Feature columns are RMS-normalized using calibration inputs. RBF lengths in
normalized input coordinates x/1.5: 0.25,0.5,1,2. Ridge and kernel length are
selected by analytic leave-one-out error on the observed residual targets.
The isotropic kernel uses four lengths; the ARD control evaluates all 64
coordinate-length combinations. Its extra search cost is recorded.

Neural solvers use L-BFGS, strong-Wolfe search, history 20. A deterministic
75%/25% calibration split selects among 30,120,480 iterations using noisy
validation error. Refit all observations from the identical initialization for
the selected budget. Report both validation fitting and refitting work. This
is not matched FLOPs with closed-form residual adapters.

Primary paired contrasts at 64 labels under the added-mechanism shift:
RBF residual minus warm full-network, and RBF residual minus learned-head fit.
These establish where output-only access is sufficient under a flexible
deployment contract. Parameter shifts and fresh neural residuals are controls.

## Development, freeze, and analysis

Use seeds 0-1 for development and freeze all settings in Git before seeds 60-71.
The 12 new seed clusters contain four systems each; average systems within a
seed and bootstrap independent seeds for descriptive 95% intervals. Report each
system separately as well. Keep negative results, all method rows, storage
counts, selected hyperparameters, source errors, and nonfinite results.

Evaluate physical acceleration RMSE, extrapolation RMSE, and 100-step rollout
RMSE. Source-only and residual wrappers get the same source function. No actual
LLM or semantic weight-reading claim is tested. No optimizer, kernel, or residual
model is claimed novel. The purpose is a stronger access comparison and a
boundary for what the eventual paper could defensibly claim.

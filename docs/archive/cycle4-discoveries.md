# Cycle 4 discoveries

The task is now adaptation from actual noisy changed-environment observations. This differs from specification editing. No mechanism library is supplied.

Under added mechanisms and 64 noisy observations, full warm adaptation achieved physical RMSE 0.0856, compared with 0.0894 for fresh full adaptation, 0.1338 for learned-head adaptation, 0.0741 for isotropic residuals, and 0.0372 for coordinate-scaled residuals. No adapter received the changed force formulas.

The primary isotropic-RBF-minus-warm-full physical-error contrast was -0.011534 [-0.021276, -0.0018148]; isotropic RBF minus learned head was -0.059743 [-0.067416, -0.052122]. The additional coordinate-scaled RBF control minus warm full was -0.048363 [-0.057862, -0.039339]. These exploratory paired intervals resample independent seed clusters and are not multiplicity-adjusted hypothesis tests. Coordinate scaling also increases hyperparameter-search work, so its contrast is not a pure causal test of one geometric factor.

For the affine parameter-shift control at 16 observations, linear-residual physical error was 0.0154, compared with 0.0556 for warm full adaptation and 0.2063 for a fresh full model. This control favors a simple correction by construction and should not be interpreted as evidence of unrestricted adaptation capability. Under added mechanisms at 64 observations, fresh full-network and fresh neural-residual mean errors were 0.0894 and 0.0675. The latter retains source predictions at deployment without copying the source weights into its correction model.

The primary isotropic residual comparison favors prediction-only reuse in this setting. The additional controls and earlier studies show why conclusions must be tied to an access and deployment contract. An initialization advantage over a fresh student does not by itself establish an advantage over methods that retain the source function. Internal feature reuse and functional reuse should be evaluated separately, especially when observed changes differ from source training dynamics.

The kernel and neural residuals deploy around a callable source function. Their performance is not equal-capacity superiority over in-place edits, and the low-dimensional state interface is favorable to classical regression. No LLM is tested.

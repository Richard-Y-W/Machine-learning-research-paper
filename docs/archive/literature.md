# Literature and novelty audit

Checked against primary sources on 2026-11-10. This is a targeted initial audit,
not a systematic review. Recent WSP reports are preliminary; their claims are
not treated as established capability results.

| Work | Relevance | What we cannot claim |
|---|---|---|
| [Brunton, Proctor & Kutz, 2016](https://doi.org/10.1073/pnas.1517384113), Discovering governing equations from data by sparse identification of nonlinear dynamical systems | Dynamics identification using a candidate-function library | Recovering oscillator coefficients by least squares is not novel. Our dense four-term fit is not a reproduction of sparse SINDy. |
| [Neyshabur, Salakhutdinov & Srebro, 2015](https://arxiv.org/abs/1506.02617), Path-SGD | Optimization designed around function-preserving rescaling | The need for scale-aware optimization is longstanding. Our normalized Jacobian is a diagnostic, not a newly claimed optimizer. |
| [Dinh et al., 2017](https://arxiv.org/abs/1703.04933), Sharp Minima Can Generalize For Deep Nets | Reparameterization changes parameter geometry without changing network behavior | ReLU scaling sensitivity by itself is not a new discovery. |
| [Lei, Schoelkopf & Posner, 2022](https://arxiv.org/abs/2206.11131), Variational Causal Dynamics | Modular world-model adaptation under interventions | Efficient adaptation to sparse mechanism shifts already exists. We do not implement or beat VCD. |
| [Zhou et al., 2023](https://arxiv.org/abs/2302.14040), Permutation Equivariant Neural Functionals | Networks that process weights with permutation structure | Weight editing and neuron-permutation equivariance are established research areas. We do not train a neural functional. |
| [Ilharco et al., 2023](https://arxiv.org/abs/2212.04089), Editing Models with Task Arithmetic | Combining parameter changes around a common model | Our jointly specified combined intervention does not demonstrate sequential edit composition or task arithmetic. |
| [Takehi et al., 2026](https://rikiyat.github.io/blog/weight-space-programming/), Sparks of Weight-Space Programming | Motivates recent interest in LLM reading/writing of weights | This pilot has no LLM editor and neither replicates nor refutes those demonstrations. |
| [Ahmed, 2026](https://arxiv.org/abs/2611.08070), Can an LLM Actually Reason and Program about Neural Network's Weights... | Exploratory audit of shortcuts in weight interpretation | Shortcut controls and skeptical evaluation are not themselves a new contribution. |

## Position of this repository

The defensible immediate contribution is an executable screening protocol, with
paired baselines and exact representation controls. Its publication value remains
unestablished. The algebraic observations in the draft are included to expose
what the benchmark already assumes; they are not claimed as novel theorems.

A strong next paper would need either a practical constraint under which weight
access provides a reproducible advantage, or a characterization beyond the
classical facts summarized here. Broader citations (hypernetworks, implicit
representations, model editing, system identification, natural-gradient methods)
are needed before claiming comprehensive coverage.

## Follow-up additions

- [Hinton, Vinyals & Dean, 2015](https://arxiv.org/abs/1503.02531) supplies the
  classical teacher/student distillation context. Our fresh-student comparator
  is ordinary supervised distillation, not a new algorithm.
- [Zhang et al., 2025, version 1](https://arxiv.org/abs/2502.08005v1) explicitly
  separates teacher supervision from initialization in one-step diffusion.
  Warm-start advantages alone therefore do not establish a new principle. The
  latest arXiv version is a revised work titled *DiffRatio*; the manuscript cites
  version 1 specifically for its initialization analysis.
- The parameter-relative Jacobian control is a diagonal-coordinate geometry
  diagnostic. No claim of first introduction, universal invariance, or a new
  natural-gradient method is made.

## Third-cycle interpretation

- [Kumar et al., 2022](https://arxiv.org/abs/2202.10054) studies how fine-tuning
  can distort pretrained features and affect generalization. Our optimization
  curves do not independently establish feature distortion as their cause.
- [Anguita et al., 2026](https://proceedings.mlr.press/v306/anguita26a.html)
  analyzes initialization-dependent fine-tuning regimes. An error gap after
  similar calibration fit is compatible with established inductive-bias ideas;
  it is not claimed as a new general theory.
- Query-placement effects are interpreted through the elementary regression
  error identity already given in the manuscript, alongside classical system
  identification [Brunton et al., 2016](https://doi.org/10.1073/pnas.1517384113).
  Neither broad probing nor discrete library selection is presented as novel.

## Functional-reuse controls

- [Rasmussen & Williams, 2006](https://gaussianprocess.org/gpml/) provides the
  standard kernel/GP context. Our residual comparator is point-prediction kernel
  ridge with calibration-only length/ridge selection, not a new GP method.
- [Brudigam et al., 2022](https://proceedings.mlr.press/v168/brudigam22a.html)
  combines learned GP residual dynamics with variational integrators. Residual
  dynamics learning is established; the current study compares practical source
  access and deployment contracts rather than claiming a new residual approach.

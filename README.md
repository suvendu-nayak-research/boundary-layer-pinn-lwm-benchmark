# Boundary-Layer PINN–LWM Benchmark

Code for a numerical benchmark comparing direct Legendre wavelet collocation with physics-informed neural networks on a singularly perturbed transport boundary-value problem.

**Authors:** Suvendu Nayak and Arun Kumar Gupta

**Current manuscript:**  
*Boundary-Layer Limitations of a Standard Soft-Boundary Physics-Informed Neural Network: A Legendre Wavelet Collocation Benchmark for a Singularly Perturbed Transport Problem*

**Original arXiv version:** [arXiv:2606.09676](https://arxiv.org/abs/2606.09676)

---

## Problem

We consider

```math
u''(x)-\mathrm{Pe}\,u'(x)=0,
\qquad 0\leq x\leq 1,
```

with

```math
u(0)=1,
\qquad
u(1)=0.
```

The exact solution is

```math
u_{\mathrm{exact}}(x)
=
\frac{1-\exp(\mathrm{Pe}(x-1))}
     {1-\exp(-\mathrm{Pe})}.
```

The four benchmark cases are

```math
\mathrm{Pe}\in\{1,10,100,1000\}.
```

As Pe increases, the solution remains close to one over most of the interval and changes rapidly near the right boundary.

---

## Repository structure

```text
boundary-layer-pinn-lwm-benchmark/
│
├── LWM_Boundary_Layer_Benchmark_Revise.ipynb
├── PINN_Standard_Soft_Boundary.ipynb
├── PINN_Robustness_Hard_Boundary.ipynb
│
├── archive/
│   └── arxiv-v1/
│       └── PINN_MultiNu_Standard_PINN_Final.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md
```

The three active notebooks are committed without stored execution output. Running them generates the numerical results, figures, and output files.

The archived notebook preserves the earlier PINN implementation associated with the original arXiv version.

---

## Methods

### Direct two-sum Legendre wavelet collocation

`LWM_Boundary_Layer_Benchmark_Revise.ipynb` implements the direct cellwise Legendre wavelet formulation.

The approximation is

```math
u_{J,M}(x)
=
\sum_{j=0}^{J-1}
\sum_{m=0}^{M-1}
c_{j,m}\psi_{j,m}(x).
```

Each cell uses `M - 2` Gauss–Legendre residual collocation points. The two boundary conditions are imposed directly as algebraic equations, and continuity of the solution and its first derivative is enforced at the internal interfaces.

The equation count is

```math
J(M-2)+2+2(J-1)=JM,
```

giving the square system

```math
A\mathbf{c}=\mathbf{b}.
```

The cells are uniformly distributed over the interval; the method does not use adaptive local mesh refinement.

The main configurations are:

| Pe | Cells J | Modes M | Unknowns |
|---:|---:|---:|---:|
| 1 | 4 | 6 | 24 |
| 10 | 8 | 6 | 48 |
| 100 | 40 | 6 | 240 |
| 1000 | 200 | 6 | 1200 |

The notebook also contains a refinement study for Pe = 1000 with

```text
J = 100, 150, 200, 250
M = 6
```

and a comparison between 101-point uniform and dense-grid error evaluation.

Running the notebook creates

```text
lwm_results/
```

### Standard soft-boundary PINN

`PINN_Standard_Soft_Boundary.ipynb` implements the baseline PINN for all four Pe values using seed `1234`.

The neural approximation is trained with the residual

```math
r_\theta(x)
=
u_\theta''(x)-\mathrm{Pe}\,u_\theta'(x),
```

and loss

```math
\mathcal{L}
=
\mathcal{L}_r
+
100\,\mathcal{L}_b.
```

The configuration is:

| Setting | Value |
|---|---|
| Hidden layers | 4 |
| Width | 64 |
| Activation | `tanh` |
| Precision | `float64` |
| Training points | 10,000 equally spaced points |
| Initialization | Xavier normal |
| Adam updates | 15,000 |
| Adam learning rate | 1e-3 |
| L-BFGS max iterations | 500 |
| L-BFGS max evaluations | 500 |
| Boundary-loss weight | 100 |
| Baseline seed | 1234 |

The boundary conditions are imposed through the loss rather than built into the trial solution.

Running the notebook creates

```text
pinn_standard_results/
```

### Robustness and hard-boundary PINN tests

`PINN_Robustness_Hard_Boundary.ipynb` contains the additional experiments for Pe = 100 and 1000.

The soft-boundary PINN is repeated with

```text
1234, 2345, 3456, 4567, 5678
```

to examine sensitivity to network initialization.

The notebook also tests the hard-boundary trial solution

```math
u_\theta^{H}(x)
=
1-x+x(1-x)N_\theta(x),
```

which satisfies both endpoint conditions exactly.

The soft- and hard-boundary PINNs are compared using seed `1234` with the same network architecture, numerical precision, collocation grid, and optimization settings.

The term **hard-boundary** refers only to this PINN formulation. The LWM already imposes the boundary conditions directly in its algebraic system.

Running the notebook creates

```text
pinn_robustness_hardbc_results/
```

---

## Evaluation

All numerical solutions are compared with the exact solution on the same dense evaluation grid.

The grid is the sorted unique union of:

- 2001 uniformly spaced points on `[0,1]`;
- 5000 points concentrated near `x = 1`;
- the two endpoints.

After removing duplicates, the grid contains **6981 points**.

The main reported error measures are the maximum absolute error,

```math
E_\infty
=
\max_i
\left|
u_{\mathrm{num}}(x_i)-u_{\mathrm{exact}}(x_i)
\right|,
```

and the L2-type error evaluated by trapezoidal integration on the ordered dense grid.

The evaluation grid is independent of both the LWM collocation points and the PINN training points.

---

## Baseline results

The main comparison uses the common dense evaluation grid.

| Pe | LWM E∞ | Soft PINN E∞ | LWM E2 | Soft PINN E2 |
|---:|---:|---:|---:|---:|
| 1 | 1.056E-09 | 5.588E-06 | 4.341E-10 | 2.997E-06 |
| 10 | 6.327E-06 | 1.843E-05 | 1.312E-06 | 9.041E-06 |
| 100 | 2.213E-04 | 5.019E-01 | 2.021E-05 | 4.949E-01 |
| 1000 | 4.735E-03 | 5.000E-01 | 2.006E-04 | 4.995E-01 |

At the selected LWM resolutions, the maximum absolute error remains below `5e-3` for all four cases.

The standard soft-boundary PINN gives small errors for Pe = 1 and 10. For Pe = 100 and 1000, the trained solution remains close to one-half over most of the interval and does not reproduce the boundary-layer profile.

The two implementations do not use matched numbers of degrees of freedom or identical solution procedures. These results therefore compare the complete formulations under the stated settings rather than equal-cost or equal-resolution computations.

---

## Five-seed robustness test

For the high-Pe soft-boundary PINN:

| Pe | Mean E∞ | Sample standard deviation |
|---:|---:|---:|
| 100 | 5.018909E-01 | 1.463498E-05 |
| 1000 | 5.000243E-01 | 4.179827E-07 |

The same nearly constant high-Pe behavior is obtained across the five tested initializations.

---

## Hard-boundary PINN test

The soft- and hard-boundary formulations give:

| Pe | Soft E∞ | Hard E∞ | Soft E2 | Hard E2 |
|---:|---:|---:|---:|---:|
| 100 | 5.018834E-01 | 3.375366E-04 | 4.948760E-01 | 1.451444E-04 |
| 1000 | 5.000250E-01 | 9.918847E-01 | 4.995006E-01 | 5.753610E-01 |

Exact boundary enforcement substantially changes the Pe = 100 result. Under the same tested training configuration, it does not recover the thin boundary layer for Pe = 1000.

The hard-boundary experiment is therefore a controlled test of boundary enforcement, not evidence that boundary enforcement alone explains the difference between the LWM and PINN results.

---

## LWM refinement

For Pe = 1000 and M = 6:

| J | Unknowns | E∞ | E2 |
|---:|---:|---:|---:|
| 100 | 600 | 5.188667E-02 | 3.230748E-03 |
| 150 | 900 | 1.394979E-02 | 6.945185E-04 |
| 200 | 1200 | 4.735234E-03 | 2.005968E-04 |
| 250 | 1500 | 1.887938E-03 | 7.055387E-05 |

Both error measures decrease over the tested sequence as the number of cells is increased.

---

## Evaluation-grid sensitivity

For the same Pe = 1000, J = 200, M = 6 LWM solution:

| Evaluation grid | Points | E∞ |
|---|---:|---:|
| Uniform | 101 | 1.504911E-05 |
| Dense | 6981 | 4.735234E-03 |

The largest error is concentrated near the right boundary. A sparse uniform evaluation grid can therefore substantially underestimate the maximum error.

---

## Reproducing the experiments

Clone the repository:

```bash
git clone https://github.com/suvendu-nayak-research/boundary-layer-pinn-lwm-benchmark.git
cd boundary-layer-pinn-lwm-benchmark
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```text
LWM_Boundary_Layer_Benchmark_Revise.ipynb
PINN_Standard_Soft_Boundary.ipynb
PINN_Robustness_Hard_Boundary.ipynb
```

The notebooks can be executed in Jupyter Notebook, JupyterLab, or Google Colab.

The active notebooks record the numerical settings and runtime environment information used during execution. Numerical timings can vary across hardware and software environments, and PINN results can show small numerical differences across PyTorch/CUDA environments.

---

## Scope

This repository benchmarks the implementations and numerical settings described above.

The standard soft-boundary PINN is a baseline formulation. The results are not a general claim that physics-informed neural networks cannot resolve boundary layers.

Adaptive sampling, alternative loss weighting, domain decomposition, and other PINN formulations are outside the scope of this benchmark.

Similarly, the LWM calculations use uniformly distributed cells; local basis support should not be interpreted as adaptive mesh refinement.

---

## Citation

For the original arXiv version:

```bibtex
@misc{nayak2026boundary,
  title         = {Boundary-Layer-Induced Failure of Standard Physics-Informed Neural Networks: A Legendre Wavelet Collocation Benchmark for Singularly Perturbed Transport Problems},
  author        = {Nayak, Suvendu and Gupta, Arun Kumar},
  year          = {2026},
  eprint        = {2606.09676},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  url           = {https://arxiv.org/abs/2606.09676}
}
```

## License

Released under the MIT License. See `LICENSE`.

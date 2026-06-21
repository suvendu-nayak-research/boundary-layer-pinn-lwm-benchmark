# Boundary-Layer PINN–LWM Benchmark

This repository contains the computational notebooks for the study:

**Boundary-Layer-Induced Failure of Standard Physics-Informed Neural Networks: A Legendre Wavelet Collocation Benchmark for Singularly Perturbed Transport Problems**

**Authors:** Suvendu Nayak and Arun Kumar Gupta

**Preprint:** [arXiv:2606.09676](https://arxiv.org/abs/2606.09676)

## Benchmark problem

We consider the steady one-dimensional advection–diffusion boundary-value problem

$$u''(x)-\mathrm{Pe}\,u'(x)=0, \qquad 0\leq x\leq 1,$$

with boundary conditions

$$u(0)=1, \qquad u(1)=0.$$

The Peclet number is related to the diffusion coefficient by

$$\mathrm{Pe}=\frac{1}{\nu}.$$

The exact solution is


$$u_{\mathrm{exact}}(x)=\frac{1-e^{\mathrm{Pe}(x-1)}}{1-e^{-\mathrm{Pe}}}.$$



The experiments use

$$\mathrm{Pe}=1,\ 10,\ 100,\ 1000.$$

## Numerical methods

### Direct two-sum Legendre wavelet collocation method

The approximation is

$$u_{J,M}(x)=\sum_{j=0}^{J-1}\sum_{m=0}^{M-1}c_{j,m}\psi_{j,m}(x).$$

The coefficients are determined from residual equations, the two boundary conditions, and continuity of $u$ and $u'$ at the internal interfaces.

For the second-order problem, the number of equations is

$$J(M-2)+2+2(J-1)=JM.$$

The resulting square linear system is

$$A\mathbf{c}=\mathbf{b}.$$

### Standard soft-boundary PINN

The neural approximation is denoted by $u_{\theta}(x)$, and the differential-equation residual is

$$r_{\theta}(x)=u_{\theta}''(x)-\mathrm{Pe}\,u_{\theta}'(x).$$

The residual loss is

$$\mathcal{L}_{r}=\frac{1}{N_r}\sum_{i=1}^{N_r}\left|r_{\theta}(x_i^r)\right|^2,$$

the boundary loss is

$$\mathcal{L}_{b}=\left|u_{\theta}(0)-1\right|^2+\left|u_{\theta}(1)\right|^2,$$

and the total loss is

$$\mathcal{L}=\mathcal{L}_{r}+\lambda_b\mathcal{L}_{b}, \qquad \lambda_b=100.$$

The baseline PINN uses four hidden layers, 64 neurons per hidden layer, `tanh` activation, 10,000 uniformly distributed collocation points, Adam optimization, and L-BFGS refinement.

This repository studies the standard soft-boundary PINN formulation only. It does not claim that all PINN variants fail for boundary-layer problems.

## Repository contents

```text
boundary-layer-pinn-lwm-benchmark/
├── LWM_final.ipynb
├── PINN_MultiNu_Standard_PINN_Final.ipynb
└── README.md
```

### `LWM_final.ipynb`

Contains the direct two-sum Legendre wavelet collocation experiments for all four Peclet numbers.

Running the notebook creates the folder:

```text
lwm_results/
```

### `PINN_MultiNu_Standard_PINN_Final.ipynb`

Contains the standard soft-boundary PINN experiments for all four Peclet numbers.

Running the notebook creates the folder:

```text
pinn_standard_results/
```

## Legendre wavelet configurations

| Peclet number | Cells $J$ | Modes per cell $M$ | Unknowns $JM$ |
|---:|---:|---:|---:|
| 1 | 4 | 6 | 24 |
| 10 | 8 | 6 | 48 |
| 100 | 40 | 6 | 240 |
| 1000 | 200 | 6 | 1200 |

## Main reported comparison

| Peclet number | LWM maximum error | PINN maximum error | LWM $L_2$-type error | PINN $L_2$-type error |
|---:|---:|---:|---:|---:|
| 1 | 1.056E-09 | 1.391E-05 | 4.341E-10 | 7.792E-06 |
| 10 | 6.327E-06 | 4.592E-06 | 1.312E-06 | 2.313E-06 |
| 100 | 2.213E-04 | 5.019E-01 | 2.021E-05 | 4.949E-01 |
| 1000 | 4.735E-03 | 5.000E-01 | 2.006E-04 | 4.995E-01 |

The Legendre wavelet method resolves all four test cases. The standard soft-boundary PINN performs well for $\mathrm{Pe}=1$ and $\mathrm{Pe}=10$, but does not resolve the sharp boundary layers for $\mathrm{Pe}=100$ and $\mathrm{Pe}=1000$.

## Error evaluation

For a numerical approximation $u_{\mathrm{num}}$, the maximum absolute error is

$$E_{\infty}=\max_i\left|u_{\mathrm{num}}(x_i)-u_{\mathrm{exact}}(x_i)\right|.$$

The $L_2$-type error is

$$E_2=\left(\int_0^1\left|u_{\mathrm{num}}(x)-u_{\mathrm{exact}}(x)\right|^2\,dx\right)^{1/2}.$$

Both notebooks use a dense evaluation grid with uniform points and additional points clustered near $x=1$, where the boundary layer develops.

## Requirements

- Python
- NumPy
- Matplotlib
- PyTorch

## Running the notebooks

Open the notebooks in Jupyter Notebook, JupyterLab, or Google Colab and run them from top to bottom.



## Reproducibility note

The computational cells and stored outputs in the notebooks correspond to the experiments reported in the manuscript.

Small numerical differences may occur when the notebooks are rerun using different hardware, PyTorch or CUDA versions, or linear-algebra libraries.

## Citation

```bibtex
@misc{nayak2026boundary,
  title         = {Boundary-Layer-Induced Failure of Standard Physics-Informed Neural Networks: A Legendre Wavelet Collocation Benchmark for Singularly Perturbed Transport Problems},
  author        = {Nayak, Suvendu and Gupta, Arun Kumar},
  year          = {2026},
  eprint        = {2606.09676},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2606.09676}
}
```

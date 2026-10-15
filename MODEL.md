# Optical formation, correlation estimation and timing derivations

Adapted from the MIT predecessor at commit 80d7628906d3bbaa29fd23d5471b5ae0f7c46148. See README for this lesson’s pilot decision and budgets.

## Optical and camera model

The pupil amplitude is `exp(-(rho_x²+rho_y²)/a²)`, with amplitude radius `a=1000 µm`, wavelength `lambda=0.550 µm`, and focal length `f=16000 µm`. This is an **untruncated Gaussian pupil**, not a uniform circular aperture or an f-number model. Scalar paraxial monochromatic Fourier optics, an aberration-free isolated point and instantaneous exposure are assumed.

The Fourier amplitude is proportional to `exp(-(pi*a*x/(lambda*f))²)`. Squaring gives intensity SD `lambda*f/(2*pi*a)`. At pixel pitch `p=4 µm`,

```text
sigma_px = lambda*f/(2*pi*a*p) = 0.350140875 px
mu_j(u) = B + F [Phi((j+0.5-u)/sigma) - Phi((j-0.5-u)/sigma)]
```

`Phi` is the standard-normal CDF; `F=10000 ADU`, `B=20 ADU`. The camera has square pixels, no distortion/skew, principal point `(0,0)` and `K=diag(4000,4000,1)` in pixels. We simulate the horizontal marginal of a separable 2-D PSF. Signed ROI pixel centers are `j=-16,...,+16`, increasing to the right; each integrates `[j-0.5,j+0.5]`. These are not top-left array coordinates.

Each marginal sample receives independent Gaussian read noise of SD 2 ADU. There is no shot noise, saturation, quantization, or photometry fitting; flux, background and optical width are known. Least squares fits `u` using the analytic pixel-integral derivative. This high-SNR toy model isolates clock uncertainty, not a full camera noise budget.

The independently known image trajectory is `u(r)=u0+v*r`, `u0=0.23 px`, `v=0.8 px/reference second`. Clocks obey `r=alpha*t+beta`: true `alpha=1.002 reference s/master s`, `beta=0.030 reference s`. A frame tagged `T=2.0 master s` was exposed at `T+delta`, with true delay `delta=0.040 master s`.

## Derive the transfer failure

For one m×2 repeat block Z with constant unknown mean, write

```text
R_t[k,l] = rho_current^|k-l|
W_h = R(rho_assumed)^(-1)
Q_h = W_h - W_h 1 (1^T W_h 1)^(-1) 1^T W_h
S = Z^T Q_h Z / (m-1).
```

Since `Q_h 1=0`, the mean vanishes. Expanding the quadratic form entrywise gives

```text
E[S | rho_assumed] = kappa * C
kappa = tr(Q_h R_t)/(m-1).
```

At the correct correlation, `tr(Q_t R_t)=m-1`, so kappa=1. After a regime change, kappa generally differs from 1 **even with an infinitely long, perfectly estimated old characterization**. More old data cannot repair a transfer assumption. Independent finite-sample estimates add another layer: average this conditional factor over the characterization estimator. Independence prevents reuse bias but not nonlinear plug-in bias.

A common scalar multiplying every event covariance cancels from the clock objective. Here the two classes have different transfer factors; their relative weights change and affect delay risk. The current lesson crosses one-class temporal correlation changes with a separate nuisance-covariance transformation; see README.

The trace factor is a theoretical diagnostic, not a feasible policy input. This lesson reports actual measurement risk rather than treating the factor as instrument accuracy.

## Derive correlation estimation from mean-free data

For one characterization sequence `Z` of shape n×p with p=2:

```text
Z ~ matrix normal(1 mu^T, R(rho), C)
W = R^-1
Q = W - W1 (1^T W1)^-1 1^T W
A(rho) = Z^T Q Z
```

Q removes the unknown constant two-clock mean. To derive a restricted likelihood, choose any `(n-1)×n` orthonormal contrast matrix D with `D1=0`. Then `Y=DZ` has row covariance `V=DRD^T`, no unknown mean, and column covariance C. Minus twice its Gaussian log likelihood, omitting constants, is

```text
p log|V| + (n-1) log|C| + tr(C^-1 Y^T V^-1 Y).
```

Differentiating with respect to C gives `C_hat=A/(n-1)`; substitution makes the trace term `(n-1)p`, constant in rho. Using `D^T V^-1 D=Q` and `|V|=|R|(1^T W1)/n`, minimize

```text
J(rho) = p log|R| + p log(1^T W1) + (n-1) log|A/(n-1)|.
```

The omitted `-p log n` has no effect on the minimum. This is a profile **restricted** likelihood, not the ordinary likelihood with n degrees of freedom, and it does not guarantee an unbiased rho estimate at finite n.

For stationary AR(1), `|R|=(1-rho²)^(n-1)`. Its inverse is tridiagonal with end diagonal 1, interior diagonal `1+rho²`, and off-diagonal `-rho`, all divided by `1-rho²`. `reml_score` uses this identity in O(n) operations. No dense inverse is required. Subtracting the ordinary column mean first improves numerical stability because Q annihilates every constant.

```python
# The sequence is the only input: no true noise parameters enter this fit.
rho_hat = fit_rho(characterization)
# Select bounded minima of J on [-.95,.95]; keep and count boundary fits.
# Profile nuisance C is not used downstream in this experiment.
```

The code evaluates 39 grid points and refines all bracketed local minima, also considering endpoints. This is a reproducible numerical procedure, not a global-optimum proof for arbitrary data. The interval `[-.95,.95]` is an estimator assumption, not a confidence interval. Negative estimates remain allowed even when true rho=0. Boundary frequencies are reported so clipping cannot silently masquerade as exact recovery.

## From calibration weights to optical delay

Treat `S_i` as a plug-in covariance, temporarily fixed during each fit. With latent true time `tau_i`, minimize

```text
Q = sum_i [z_i-(tau_i,alpha*tau_i+beta)]^T S_i^-1
          [z_i-(tau_i,alpha*tau_i+beta)]
```

For `d=(1,alpha)`, differentiation gives

```text
tau_i* = d^T S_i^-1 (x_i,y_i-beta) / (d^T S_i^-1 d).
V_i(alpha) = alpha²*S_i,xx - 2*alpha*S_i,xy + S_i,yy
Q_profile = sum_i (y_i-alpha*x_i-beta)²/V_i(alpha)
beta*(alpha) = sum_i [(y_i-alpha*x_i)/V_i] / sum_i [1/V_i].
```

Search numerically over `alpha in [0.5,1.5]`. The implementation checks a 65-point grid, refines bracketed minima and rejects boundary/flat fits. This is not a global-optimum guarantee for arbitrary data. Independent full latent-event optimization validates the implemented objective. There is no `log V_i` term: this profiles fixed latent times; it does not marginalize random times. With estimated covariance, this is a **plug-in procedure**, not the full likelihood jointly accounting for covariance uncertainty. We make no confidence-interval or coverage claim.

Finally,

```text
r_hat = (u_hat-u0)/v
 delta_hat = (r_hat-beta_hat)/alpha_hat - T
 d(delta) ≈ du/(v*alpha) - d(beta)/alpha - (T+delta)*d(alpha)/alpha.
```

Thus precise optical localization can coexist with poor delay precision when calibration weights are uncertain and rate errors are extrapolated.

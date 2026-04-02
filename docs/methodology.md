# Methodology: Bayesian Marketing Mix Model

This document details the mathematical framework behind the Bayesian Marketing Mix Model, covering model formulation, Bayesian estimation, baseline decomposition, and uncertainty-aware budget optimization. The model is implemented using [PyMC-Marketing](https://www.pymc-marketing.io/), a probabilistic programming library purpose-built for media mix modeling.

---

## 1. Model Formulation

### 1.1 Revenue Equation

Total weekly revenue is modeled as the sum of a baseline component and media contributions:

```
Revenue(t) = Baseline(t) + Σ_c MediaContribution_c(t) + ε(t)
```

where `c ∈ {Paid Search, Paid Social, Display, Email, TV/OTT, Affiliate}`, and `ε(t) ~ Normal(0, σ²)`.

This top-level structure is identical to a classical MMM. The critical difference lies in *how* the transform parameters, response coefficients, and noise variance are estimated — jointly, probabilistically, and with principled uncertainty quantification.

### 1.2 Saturation Transform

Each channel's spend exhibits diminishing returns, modeled with PyMC-Marketing's `LogisticSaturation`:

```
Sat(x, λ) = 1 − exp(−λx)
```

where `λ` is a channel-specific saturation rate parameter **learned from data** (not hardcoded). This function maps `[0, ∞)` to `[0, 1)`:

- At `x = 0`: `Sat = 0` (no response)
- At `x = 1/λ`: `Sat ≈ 0.632` (63.2% of maximum response)
- As `x → ∞`: `Sat → 1` (full saturation)

The marginal response at any spend level is:

```
∂Sat/∂x = λ · exp(−λx)
```

The key departure from the OLS approach: `λ` is no longer a fixed constant chosen via grid search. It carries a prior distribution `λ ~ Gamma(3, 1)` and is estimated jointly with all other model parameters during MCMC sampling. This means the model's uncertainty about the *shape* of the response curve propagates into every downstream quantity — contribution estimates, ROI, and optimal allocations.

### 1.3 Geometric Adstock

Media spend has carryover effects — this week's ad exposure continues to influence purchasing behavior in future weeks. This is modeled with PyMC-Marketing's `GeometricAdstock`:

```
Adstock(t) = Σ_{l=0}^{l_max} α^l · Spend(t − l)   (normalized to sum to 1)
```

where:

- `α` = decay rate ∈ [0, 1), **learned per channel** (prior: `Beta(1, 3)`)
- `l_max = 10` = maximum lag window (weeks)
- `normalize = True` = adstock weights sum to 1, preserving the scale of spend

The `Beta(1, 3)` prior on α places more mass near zero, encoding a mild expectation that most media effects decay quickly while still allowing the data to push toward longer carryover when the evidence supports it. The effective half-life under geometric decay is:

```
Half-life = −ln(2) / ln(α)
```

Unlike the OLS specification where each channel's lag and decay were fixed *a priori*, the Bayesian model learns decay rates from data. The posterior distribution on α directly quantifies how confident the model is in each channel's carryover window.

### 1.4 Full Model Specification

Combining saturation, adstock, controls, and seasonality, the full model is:

```
y_scaled(t) = intercept
             + Σ_c β_c · Sat(Adstock(x_c(t), α_c), λ_c)
             + γ_nov · 𝟙(Nov) + γ_dec · 𝟙(Dec) + γ_trend · t̃
             + Σ_{m=1}^{2} [γ_sin_m · sin(2πmt/52) + γ_cos_m · cos(2πmt/52)]
             + ε(t)
```

where:

- `y_scaled` = MaxAbsScaler-transformed weekly revenue
- `intercept` = baseline revenue floor
- `β_c` = channel response coefficient (revenue contribution at full saturation)
- `Sat(Adstock(·))` = composed saturation-over-adstock transform with learned parameters
- `γ_nov, γ_dec` = binary holiday control coefficients
- `γ_trend · t̃` = linear trend (t̃ normalized to [0, 1])
- `Σ Fourier` = 2-mode annual seasonality (4 Fourier terms at periods 52 and 26 weeks), configured via PyMC-Marketing's `yearly_seasonality=2`
- `ε(t) ~ Normal(0, σ²)` = observation noise

All channel spend variables `x_c` and the target `y` are scaled using `MaxAbsScaler`, handled internally by PyMC-Marketing's `MMM` class. This ensures numerically stable MCMC sampling without manual preprocessing.

---

## 2. Bayesian Estimation

### 2.1 Why Bayesian Over OLS

The previous OLS implementation used a two-phase approach: grid search for transform parameters (k, lag, λ), then closed-form regression for coefficients (β). While computationally fast, this approach has fundamental limitations that Bayesian estimation resolves:

| Limitation of OLS | Bayesian Solution |
|---|---|
| No uncertainty on transform parameters — k and λ were point estimates from grid search | α, λ estimated jointly with β; full posterior distributions on all parameters |
| Negative coefficients possible (required post-hoc clamping) | `HalfNormal` priors on β_c naturally constrain media effects to be non-negative |
| Transform parameters disconnected from coefficient estimation | Joint estimation means uncertainty in α and λ propagates correctly into β and downstream metrics |
| Point estimates only — no confidence intervals on ROI or optimal budget | Posterior distributions enable probabilistic optimization with credible intervals |

In short: every number the model produces now comes with a calibrated uncertainty estimate, and the model's structure encodes domain knowledge (media effects are non-negative, decay is usually fast) without sacrificing flexibility.

### 2.2 Prior Specification

Priors encode domain knowledge while remaining flexible enough for the data to dominate:

**Adstock decay:**
```
α_c ~ Beta(1, 3)    per channel
```
Weakly favors lower decay (faster carryover dissipation). Mean = 0.25, allowing the posterior to concentrate anywhere in [0, 1) if the data warrants it.

**Saturation rate:**
```
λ_c ~ Gamma(3, 1)    per channel
```
Weakly informative with mean = 3. Permits a wide range of saturation curve shapes, from near-linear (λ ≈ 0) to rapidly saturating (λ >> 1).

**Channel response coefficients:**
```
β_c ~ HalfNormal(σ = n_channels × spend_share_c)    per channel
```
The `HalfNormal` support on [0, ∞) enforces non-negative media effects by construction — no post-hoc clamping needed. The scale is informed by each channel's share of total spend, encoding the reasonable prior that channels receiving more budget are likely responsible for proportionally more revenue.

**Intercept and controls:**
```
intercept ~ Normal(0, 2)
γ_control ~ Normal(0, 2)       (trend, is_nov, is_dec)
γ_fourier ~ Laplace(0, 1)      (Fourier seasonality terms)
```
The `Laplace` prior on Fourier terms provides mild L1 regularization, encouraging sparse seasonality — the model can shrink unneeded harmonic modes toward zero rather than overfitting to noise.

**Likelihood:**
```
y_scaled(t) ~ Normal(μ(t), σ)
σ ~ HalfNormal(2)
```

### 2.3 MCMC Sampling

The posterior is approximated via the No-U-Turn Sampler (NUTS), an adaptive variant of Hamiltonian Monte Carlo:

| Setting | Value | Rationale |
|---|---|---|
| Chains | 4 | Enables R-hat convergence diagnostics |
| Warmup draws | 1,500 | Sufficient for NUTS adaptation (step size, mass matrix) |
| Posterior draws | 1,000 | 4,000 total draws provides stable posterior summaries |
| `target_accept` | 0.9 | Higher acceptance reduces divergences in funnel geometries |

Total computational cost: ~1–3 minutes on a modern CPU (vs. < 1 second for OLS). This is a one-time cost that buys calibrated uncertainty across all downstream analyses.

### 2.4 Data Scaling

PyMC-Marketing's `MMM` class applies `MaxAbsScaler` internally to both channel spend variables and the target. This maps all values to [−1, 1], ensuring:

- NUTS operates on a well-conditioned posterior geometry
- Prior scales are interpretable (priors are set in scaled space)
- Posterior means are automatically inverse-transformed for reporting

### 2.5 Convergence Diagnostics

Every model fit is validated against four diagnostic criteria:

1. **R-hat < 1.01** for all parameters — confirms chains have converged to the same stationary distribution
2. **Zero divergences** — divergent transitions indicate the sampler failed to explore some region of the posterior, biasing estimates
3. **Effective sample size (ESS) > 400** — ensures posterior summaries are reliable (bulk ESS for means, tail ESS for intervals)
4. **Posterior predictive checks** — simulated data from the posterior should visually and statistically resemble observed data (calibration, coverage)

---

## 3. Baseline Decomposition

### 3.1 Definition

The baseline is the model's predicted revenue when all media spend is set to zero:

```
Baseline(t) = intercept + γ_trend · t̃ + γ_nov · 𝟙(Nov) + γ_dec · 𝟙(Dec)
             + Σ_{m=1}^{2} [γ_sin_m · sin(2πmt/52) + γ_cos_m · cos(2πmt/52)]
```

This is a **model output**, not an input assumption. It represents the revenue that would exist from organic demand, brand equity, seasonality, and macro trends even without any paid media.

### 3.2 Components

| Component | Description |
|---|---|
| Intercept | Constant revenue floor — the minimum organic demand |
| Trend | Linear growth/decline over the observation window |
| Seasonality (Fourier) | Annual cyclicality captured by 2-mode Fourier terms (52-week and 26-week periods) |
| Holiday indicators | Discrete lifts for November (pre-holiday) and December (peak holiday) |

### 3.3 Credible Intervals

Because every coefficient is a posterior distribution, the baseline itself is a distribution at each time point. Reporting includes:

- **Posterior mean** baseline trajectory
- **90% highest density interval (HDI)** — the narrowest interval containing 90% of posterior mass
- **Baseline share of total revenue** with uncertainty bounds

This answers a question OLS could not: *How confident are we in the baseline level?* If the HDI is wide, the split between organic and paid revenue is uncertain — a critical caveat for strategic decisions about media dependency.

---

## 4. Budget Optimization

### 4.1 Objective

Maximize total media-attributable revenue subject to a fixed budget constraint:

```
max     Σ_c β_c · Sat(Adstock(spend_c, α_c), λ_c)
s.t.    Σ_c spend_c = B          (total budget)
        spend_c ≥ f_c · B        (minimum floor per channel)
```

where `B` is the total weekly budget and `f_c` is the minimum allocation fraction for channel c.

### 4.2 Optimality Condition

At the optimum, the marginal ROI is equalized across all unconstrained channels:

```
β_c · λ_c · exp(−λ_c · Adstock(spend_c)) = μ    for all c not at floor
```

where `μ` is the shadow price of the budget constraint. Channels at their minimum floor have marginal ROI below `μ`.

### 4.3 Learned Response Functions

The response function for each channel uses **posterior mean** estimates of α, λ, and β:

```
Response_c(spend) = β̂_c · Sat(Adstock(spend, α̂_c), λ̂_c)
```

These parameters were learned jointly from data — not grid-searched or hardcoded. This means the response curves reflect what the data actually supports, regularized by domain-informed priors.

### 4.4 Uncertainty-Aware Optimization

The optimizer is run not just on posterior means, but across posterior samples to produce:

1. **Lift distribution** — run the allocation optimizer on each posterior draw, producing a distribution of expected lift values
2. **90% credible interval on lift** — the range within which the true lift falls with 90% probability
3. **P(lift > 0)** — the posterior probability that the optimized allocation outperforms the current allocation

This is the decisive advantage over deterministic optimization: stakeholders can see not just "the model recommends X" but "the model is 94% confident that X outperforms the status quo, with expected lift between 8% and 31%."

### 4.5 Minimum Spend Floors

Floors prevent unrealistic allocations and reflect business constraints:

| Channel | Min Floor | Rationale |
|---|---|---|
| Paid Search | 22% | Protects branded search defense |
| Paid Social | 16% | Maintains always-on prospecting |
| Display | 7% | Retargeting pipeline |
| Email | 4% | Owned channel, low base cost |
| TV / OTT | 22% | Brand building minimum viable presence |
| Affiliate | 6% | Partnership commitments |

---

## 5. Advantages Over OLS Approach

### 5.1 No Negative Coefficient Hack

The OLS model could produce negative channel coefficients (implying that spending *more* on a channel *reduces* revenue), which were clamped to zero post-hoc. The Bayesian model uses `HalfNormal` priors on β_c, constraining media effects to be non-negative by construction. The prior is not a hack — it encodes the defensible assumption that paid media does not destroy revenue.

### 5.2 Hyperparameters Learned From Data

Under OLS, adstock decay (λ) and saturation constants (k) were selected via grid search — a discrete, disconnected optimization that ignores uncertainty. The Bayesian model estimates α and λ jointly with all other parameters, meaning:

- The best-fit decay rate accounts for the response coefficient magnitude (and vice versa)
- Uncertainty in the saturation shape propagates into contribution estimates
- No arbitrary grid resolution limits the parameter space

### 5.3 Full Uncertainty on Every Metric

Every quantity the model produces — channel contributions, ROI, optimal allocations, baseline share — is a distribution, not a point estimate. This enables:

- **Credible intervals** on ROI rankings (Is Email really higher-ROI than Affiliate, or is it within noise?)
- **Probabilistic budget recommendations** (P(lift > 0) gives stakeholders a decision-grade confidence level)
- **Sensitivity analysis for free** (the posterior width reveals which parameters the model is least certain about)

### 5.4 Robust to Multicollinearity

Media channels are often correlated (campaigns run simultaneously, budgets scale together). OLS is notoriously sensitive to multicollinearity — coefficient estimates become unstable, standard errors inflate, and signs can flip. Bayesian priors act as principled regularization, anchoring estimates toward domain-reasonable values while letting the data pull them away when evidence is strong.

---

## 6. Limitations and Recommended Validation

### 6.1 TV/OTT Undervaluation

The model likely underestimates TV/OTT's true contribution due to:

- **Brand halo:** TV drives branded search queries attributed to Paid Search
- **Carryover truncation:** Even with learned adstock decay, geometric decay captures localized carryover; real awareness may persist for months
- **Baseline absorption:** TV-driven organic demand is partially captured by the intercept and trend, not the TV coefficient

**Recommended validation:** Run a geo-holdout test (dark TV in 2–3 DMAs for 8 weeks) and compare total market revenue — not just TV-attributed revenue.

### 6.2 Cross-Channel Interactions

The model assumes channel independence (additive contributions). In reality, channels interact: TV builds awareness that Search captures, Social engagement primes Email opens, Display retargets across the funnel. Cutting one channel may degrade another's apparent performance.

**Recommended approach:** Phase budget changes gradually (10–15% per sprint), monitoring cross-channel metrics (branded search volume, email open rates, direct traffic) as leading indicators of halo degradation.

### 6.3 Synthetic Data

This analysis uses simulated data with known ground-truth parameters. A production implementation would require:

- Actual spend and revenue data (ideally with geo-level variation)
- External validation via incrementality tests (geo-holdout, lift studies)
- Out-of-sample testing (train on 80 weeks, validate on 24)
- Prior sensitivity analysis (run the model under alternative prior specifications to assess robustness)

### 6.4 MCMC Computational Cost

Bayesian estimation via NUTS requires ~1–3 minutes per model fit (vs. < 1 second for OLS). This is negligible for a weekly reporting cadence but relevant for:

- **Rapid iteration during model development** — consider using variational inference (`pm.fit()`) as a fast approximation during exploration, then switching to full MCMC for final estimates
- **Large-scale grid searches** over model specifications — Bayesian model comparison (LOO-CV, WAIC) replaces brute-force grid search but still requires fitting each candidate model
- **Real-time optimization dashboards** — pre-compute posterior samples and cache response curves; optimization itself is fast once posteriors are available

# Methodology: Mathematical Specification

This document details the mathematical framework behind the Marketing Mix Model, covering model formulation, parameter estimation, baseline decomposition, and budget optimization.

---

## 1. Model Formulation

### 1.1 Revenue Equation

Total weekly revenue is modeled as the sum of a baseline component and media contributions:

```
Revenue(t) = Baseline(t) + Σ_c MediaContribution_c(t) + ε(t)
```

where `c ∈ {Paid Search, Paid Social, Display, Email, TV/OTT, Affiliate}`, and `ε(t) ~ N(0, σ²)`.

### 1.2 Saturation Transform

Each channel's spend exhibits diminishing returns, modeled with an exponential saturation function:

```
Sat(spend, k) = 1 − exp(−spend / k)
```

where `k` is the channel-specific half-saturation constant. This function maps `[0, ∞)` to `[0, 1)`:

- At `spend = 0`: `Sat = 0` (no response)
- At `spend = k`: `Sat ≈ 0.632` (63.2% of maximum response)
- As `spend → ∞`: `Sat → 1` (full saturation)

The marginal response at any spend level is:

```
∂Sat/∂spend = (1/k) · exp(−spend / k)
```

This monotonically decreasing derivative is the mathematical basis for diminishing returns — each additional dollar of spend produces less incremental response than the previous one.

**Why this functional form?** The exponential saturation is the most common specification in industry MMMs (used by Analytic Partners' GPS-E, and as the default in both Meta Robyn and Google Meridian). Alternatives include the Hill function `Sat(x) = x^α / (x^α + k^α)` (used in PyMC-Marketing) which offers more flexibility in curve shape via the shape parameter α, at the cost of an additional parameter to estimate.

### 1.3 Geometric Adstock

Media spend has carryover effects — today's ad exposure continues to influence purchasing behavior in future weeks. This is modeled with geometric adstock:

```
Adstock(t) = Spend(t − lag) + λ · Adstock(t − 1)
```

where:

- `lag` = channel-specific lag (weeks before peak effect). TV/OTT has the longest lag (2 weeks) due to awareness-to-action conversion time.
- `λ` = decay rate ∈ [0, 1). Higher values mean longer-lasting effects.

The effective half-life of a media exposure under geometric decay is:

```
Half-life = −ln(2) / ln(λ)
```

| Channel | Lag | Decay (λ) | Half-life |
|---|---|---|---|
| Paid Search | 0 weeks | 0.10 | 0.3 weeks |
| Paid Social | 0 weeks | 0.15 | 0.4 weeks |
| Display | 1 week | 0.30 | 0.8 weeks |
| Email | 0 weeks | 0.05 | 0.1 weeks |
| TV / OTT | 2 weeks | 0.35 | 1.0 weeks |
| Affiliate | 0 weeks | 0.08 | 0.2 weeks |

### 1.4 Full Model Specification

Combining saturation and adstock, the full regression model is:

```
log(Revenue(t)) ≈ β₀ + β_trend · t̃ + β_sin · sin(2πt/52) + β_cos · cos(2πt/52)
                   + β_nov · 𝟙(Nov) + β_dec · 𝟙(Dec)
                   + Σ_c β_c · Sat(Adstock_c(t), k_c)
```

where:

- `β₀` = intercept (baseline floor)
- `β_trend · t̃` = linear trend (t̃ normalized to [0, 1] over 104 weeks)
- `β_sin, β_cos` = annual seasonality (Fourier terms at 52-week period)
- `β_nov, β_dec` = holiday indicator variables
- `β_c` = channel response coefficient (revenue at full saturation)
- `Sat(Adstock_c(t), k_c)` = transformed spend for channel c

In practice, the estimation is performed in the linear domain on the transformed features (OLS), not as a non-linear optimization. The non-linearity is handled by the pre-regression transforms (saturation + adstock), keeping the estimation stage interpretable and stable.

---

## 2. Parameter Estimation

### 2.1 Two-Phase Approach

**Phase 1: Transform parameters (k, lag, λ) — Grid search**

Saturation constants and adstock parameters are estimated via grid search over the response surface. For each candidate parameter set, the transformed features are constructed and the model R² is evaluated. This is equivalent to profile likelihood estimation.

**Phase 2: Response coefficients (β) — OLS**

Given the optimal transforms, channel coefficients and control variable weights are estimated via ordinary least squares:

```
β̂ = (X'X)⁻¹ X'y
```

where `X` is the design matrix of transformed features and `y` is the revenue vector.

**Model fit:** R² = 0.993, indicating 99.3% of weekly revenue variance is explained by the model.

### 2.2 Why OLS Over Bayesian?

For this dataset (104 observations, 12 parameters), OLS is preferred over full Bayesian MCMC for three reasons:

1. **Sample size:** 104 weeks is sufficient for OLS but marginal for Bayesian estimation without strong priors. Weakly informative priors would dominate the posterior, making results prior-dependent.
2. **Interpretability:** OLS coefficients have a direct interpretation as marginal effects. Bayesian posteriors require additional summarization (credible intervals, posterior predictive checks).
3. **Transparency:** For a stakeholder-facing analysis, the ability to say "each additional dollar of Email spend at current levels generates $X in revenue" is more actionable than probability distributions.

A production implementation with more data (200+ weeks, geo-level variation) would benefit from Bayesian methods — specifically PyMC-Marketing or Google Meridian, which provide uncertainty quantification and principled regularization.

---

## 3. Baseline Decomposition

### 3.1 Definition

The baseline is the model's predicted revenue when all media spend is set to zero:

```
Baseline(t) = β₀ + β_trend · t̃ + β_sin · sin(2πt/52) + β_cos · cos(2πt/52)
              + β_nov · 𝟙(Nov) + β_dec · 𝟙(Dec)
```

This is a **model output**, not an input assumption. It represents the revenue that would exist from organic demand, brand equity, seasonality, and macro trends even without any paid media.

### 3.2 Components

| Component | Coefficient | Interpretation |
|---|---|---|
| Intercept (β₀) | $108,122/week | Constant revenue floor |
| Trend (β_trend) | $25,875 total growth | Linear growth over 2 years |
| Seasonality (sin/cos) | ±$12K amplitude | Annual cyclicality |
| November (β_nov) | +$60,399 | Holiday shopping lift |
| December (β_dec) | +$81,731 | Peak holiday season |

**Average baseline:** $132,779/week (16.2% of total predicted revenue)

### 3.3 Why This Matters

The baseline share indicates media dependency. At 16.2%, ShopNova is heavily media-dependent — 83.8% of revenue is directly attributable to paid channels. This means:

- Budget cuts will impact topline disproportionately
- Organic growth investments (SEO, content, CRM) should be prioritized to increase baseline resilience
- The healthy upward trend ($108K → $134K over 2 years) suggests growing brand equity

---

## 4. Budget Optimization

### 4.1 Objective

Maximize total media-attributable revenue subject to a fixed budget constraint:

```
max     Σ_c β_c · Sat(spend_c, k_c)
s.t.    Σ_c spend_c = B          (total budget)
        spend_c ≥ f_c · B        (minimum floor per channel)
```

where `B = $175,000/week` and `f_c` is the minimum allocation fraction for channel c.

### 4.2 Optimality Condition

At the optimum, the marginal ROI is equalized across all channels that are not at their floor:

```
∂Response_c / ∂spend_c = (β_c / k_c) · exp(−spend_c / k_c) = μ    for all c not at floor
```

where `μ` is the shadow price of the budget constraint. Channels at their minimum floor have marginal ROI below `μ` — they would receive less budget if the floor were removed.

### 4.3 Algorithm

The optimizer uses a greedy marginal allocation approach:

1. Initialize all channels at their minimum floor: `spend_c = f_c · B`
2. Compute remaining budget: `R = B − Σ_c f_c · B`
3. Divide `R` into 800 incremental steps
4. For each step, allocate to the channel with the highest current marginal return
5. Repeat until budget is exhausted

This greedy approach converges to the equi-marginal optimum because the saturation function is concave — marginal returns are strictly decreasing.

### 4.4 Minimum Spend Floors

Floors prevent unrealistic allocations and reflect business constraints:

| Channel | Min Floor | Rationale |
|---|---|---|
| Paid Search | 22% | Protects branded search defense |
| Paid Social | 16% | Maintains always-on prospecting |
| Display | 7% | Retargeting pipeline |
| Email | 4% | Owned channel, low base cost |
| TV / OTT | 22% | Brand building minimum viable presence |
| Affiliate | 6% | Partnership commitments |

### 4.5 Results

The optimization projects a **+24.4% lift** in media-attributable revenue ($534,821 → $665,465/week) through reallocation alone. The primary drivers:

- **Email:** 4% → 17% of budget (+264%). At 12.8x ROI and 38% saturation, Email has the highest marginal return.
- **Affiliate:** 7% → 16% of budget (+138%). At 7.2x ROI and 36% saturation, strong scaling headroom.
- **TV/OTT:** 31% → 22% of budget (−30%). Hits the minimum floor — see limitations below.

---

## 5. Limitations and Recommended Validation

### 5.1 TV/OTT Undervaluation

The model likely underestimates TV/OTT's true contribution due to:

- **Brand halo:** TV drives branded search queries attributed to Paid Search
- **Carryover truncation:** Geometric adstock with 35% decay captures ~2 weeks of effect; real awareness may persist for months
- **Baseline absorption:** TV-driven organic demand is captured by the model's intercept and trend, not the TV coefficient

**Recommended validation:** Run a geo-holdout test (dark TV in 2-3 DMAs for 8 weeks) and compare total market revenue — not just TV-attributed revenue.

### 5.2 Cross-Channel Interactions

The model assumes channel independence (additive contributions). In reality, channels interact: TV builds awareness that Search captures, Social engagement primes Email opens, Display retargets across the funnel. Cutting one channel may degrade another's apparent performance.

**Recommended approach:** Phase budget changes gradually (10-15% per sprint), monitoring cross-channel metrics (branded search volume, email open rates, direct traffic) as leading indicators of halo degradation.

### 5.3 Synthetic Data

This analysis uses simulated data with known ground-truth parameters. A production implementation would require:

- Actual spend and revenue data (ideally with geo-level variation for Meridian)
- External validation via incrementality tests (geo-holdout, lift studies)
- Out-of-sample testing (train on 80 weeks, validate on 24)
- Sensitivity analysis on saturation constants and adstock parameters

# Marketing Mix Model: Bayesian Media Budget Optimization

A full-stack Bayesian Marketing Mix Model (MMM) for a simulated DTC retail e-commerce brand (**ShopNova**), built with **PyMC-Marketing** and an interactive React dashboard. The model learns channel-level adstock, saturation, and contribution parameters from data via MCMC sampling — producing uncertainty-aware budget recommendations with **credible intervals on every metric**.

**Live Dashboard:** [View on GitHub Pages →](https://freena22.github.io/marketing-mix-model/)

---

## What This Project Demonstrates

This project replicates the kind of analysis a marketing analytics team would build internally or with a measurement partner. It covers the full workflow from data generation through Bayesian model fitting, posterior decomposition, uncertainty-aware budget optimization, and stakeholder-ready visualization.

**Technical scope:**

- Bayesian MMM via PyMC-Marketing with GeometricAdstock and LogisticSaturation transforms
- All parameters (adstock decay, saturation curves, channel betas) learned from data — no hardcoded values
- MCMC sampling: 4 chains × 1,500 tuning + 1,000 draws, target_accept=0.9
- 90% credible intervals (Bayesian HDI) on every reported metric
- Uncertainty-aware budget optimization reporting P(lift > 0) and lift CIs

**Business scope:**

- 104 weeks of weekly channel-level spend and revenue across 6 paid media channels
- Posterior-derived saturation curves, carryover effects, and response coefficients per channel
- Actionable reallocation recommendations with probabilistic confidence levels
- Nuanced treatment of upper-funnel channels (TV/OTT) and cross-channel halo effects

---

## Project Structure

```
marketing-mix-model/
├── model/
│   ├── mmm_retail.py                  # Original OLS pipeline (kept for reference)
│   └── mmm_retail_bayesian.py         # Bayesian MMM: data gen → fit → optimize → export
├── dashboard/
│   └── marketing_budget_optimizer.jsx  # Interactive React dashboard (6 tabs)
├── data/
│   └── dashboard_data.json            # Exported model results for dashboard
├── docs/
│   ├── methodology.md                 # Mathematical specification
│   └── index.html                     # Static standalone dashboard
├── requirements.txt
└── README.md
```

---

## Model Overview

### Data

| Parameter | Value |
|---|---|
| Time period | Jan 2024 – Dec 2025 (104 weeks) |
| Paid channels | Paid Search, Paid Social, Display, Email, TV/OTT, Affiliate |
| Weekly budget | ~$175,000 |
| Total revenue | ~$85M |
| Total media spend | ~$19M |
| Overall ROAS | ~4.4x |

### Approach

**Model specification:** Bayesian MMM via PyMC-Marketing with two channel-level transforms estimated jointly:

1. **Adstock (GeometricAdstock):** Channel-specific decay α learned from data — models carryover effects where spend in week *t* continues generating revenue in subsequent weeks
2. **Saturation (LogisticSaturation):** Channel-specific λ learned from data — captures diminishing returns as spend increases within each channel
3. **Controls:** Linear trend, 2-mode Fourier seasonality (annual + semi-annual), and Nov/Dec holiday indicators to isolate media impact from organic patterns

**Priors:**

| Parameter | Prior | Rationale |
|---|---|---|
| Adstock α | Beta(1, 3) | Favors moderate-to-low carryover; most media effects decay within 2–3 weeks |
| Saturation λ | Gamma(3, 1) | Weakly informative; allows data to determine inflection points |
| Channel β | HalfNormal(σ = spend share) | Scales prior width to each channel's budget proportion |

**Inference:** MCMC via PyMC (NUTS sampler) — 4 chains × 1,500 tuning steps + 1,000 posterior draws, target_accept=0.9 for stable sampling geometry.

**Baseline decomposition:** The baseline (revenue attributable to non-media factors) is extracted from the posterior distributions of control variable coefficients: `Baseline = Intercept + Trend + Seasonality + Holiday`. This is a model output, not an input assumption — consistent with the decomposition approach used by Meta Robyn, Google Meridian, and PyMC-Marketing.

**Optimization:** Uncertainty-aware marginal ROI equalization. The optimizer reports not just point-estimate lift but also 90% credible intervals on projected revenue change and P(lift > 0) for each reallocation scenario.

### Key Results

Results follow the same directional pattern as the earlier OLS model, but every metric now carries a posterior distribution rather than a point estimate:

- Email and Affiliate remain the highest-ROI channels with the most headroom for increased investment
- Paid Search and Paid Social show moderate saturation — candidates for reallocation
- TV/OTT returns the lowest measured ROI, though structural undervaluation of upper-funnel channels is a known MMM limitation
- Optimization lift is reported with credible intervals, giving stakeholders a probabilistic view of upside and downside scenarios

### Important Caveats

The model has known limitations that would need to be addressed in a production setting:

- **TV/OTT undervaluation:** MMMs structurally undercount upper-funnel channels. TV drives branded search volume and direct traffic that gets attributed elsewhere. Geo-holdout tests are needed to validate.
- **Cross-channel halo effects:** Channels interact (TV → Search intent, Social → Email engagement), but the model treats them independently. Budget cuts should be phased gradually and monitored for cross-channel degradation.
- **Synthetic data:** The dataset is simulated with known ground-truth parameters. A real implementation would use actual spend/revenue data and require additional validation (out-of-sample testing, posterior predictive checks).
- **Posterior convergence:** While the MCMC configuration (4 chains, high target_accept) is designed for reliable convergence, production use should verify R-hat < 1.01 and adequate effective sample sizes across all parameters.

---

## Dashboard

The interactive dashboard is built with React and Recharts, designed with a muted, consulting-grade aesthetic.

### Tabs

1. **Executive Briefing** — Business context, methodology, model specification rationale, and key takeaways
2. **Overview** — Revenue vs. spend trends, channel attribution, and ROI comparison
3. **MMM Results** — Revenue decomposition time series, per-channel diminishing returns curves, and performance matrix
4. **Budget Optimizer** — Side-by-side comparison of current vs. optimized allocations across 4 scenarios
5. **Scenario Simulator** — Interactive sliders for custom budget allocation with real-time projected lift
6. **AI Insights** — Generated executive summary, channel-specific findings with recommended actions, and methodology notes

---

## How to Run

### Model (Python)

```bash
pip install -r requirements.txt
python model/mmm_retail_bayesian.py
```

First run takes **1–3 minutes** (MCMC sampling). Outputs `data/dashboard_data.json` with all model results including posterior summaries and credible intervals.

The original OLS pipeline is preserved at `model/mmm_retail.py` for reference and comparison.

### Dashboard (React)

The dashboard is a single `.jsx` file designed for rendering in any React environment. Dependencies: `react`, `recharts`.

To run locally:

```bash
npx create-react-app mmm-dashboard
cd mmm-dashboard
npm install recharts
# Copy marketing_budget_optimizer.jsx to src/App.jsx (replace default)
npm start
```

Or deploy as a static page on GitHub Pages (instructions in docs).

---

## Industry Context

This project uses **PyMC-Marketing** — one of three widely-adopted open-source MMM frameworks:

| Framework | Organization | Approach | Best For |
|---|---|---|---|
| [Robyn](https://github.com/facebookexperimental/Robyn) | Meta | Ridge + Nevergrad | Rapid iteration, automated hyperparameter tuning |
| [Meridian](https://github.com/google/meridian) | Google | Bayesian + TFP | Geo-level data, uncertainty quantification |
| [**PyMC-Marketing**](https://github.com/pymc-labs/pymc-marketing) | PyMC Labs | **Bayesian + PyMC** | **Flexible custom extensions, largest community** |

The implementation here builds on PyMC-Marketing's MMM primitives (GeometricAdstock, LogisticSaturation) while adding custom priors scaled to channel spend share, structured controls, and an uncertainty-aware optimization layer that propagates posterior uncertainty through to budget recommendations.

---

## Author

**Freena Wang** — Senior Marketing & Product Analytics Professional

Built as a portfolio project demonstrating end-to-end Bayesian marketing measurement capabilities: from probabilistic modeling and MCMC inference through uncertainty-aware optimization and interactive stakeholder communication.

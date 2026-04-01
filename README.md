# Marketing Mix Model: Media Budget Optimization

A full-stack Marketing Mix Model (MMM) for a simulated DTC retail e-commerce brand (**ShopNova**), built from scratch in Python with an interactive React dashboard. The model estimates channel-level incremental contribution, identifies diminishing returns, and optimizes budget allocation — projecting a **+24.4% revenue lift** without increasing total spend.

**Live Dashboard:** [View on GitHub Pages →](#) *(coming soon)*

---

## What This Project Demonstrates

This project replicates the kind of analysis a marketing analytics team would build internally or with a measurement partner. It covers the full workflow from data generation through model fitting, baseline decomposition, budget optimization, and stakeholder-ready visualization.

**Technical scope:**

- Log-linear regression with saturation transforms and geometric adstock
- Baseline decomposition extracted from model coefficients (not assumed)
- Marginal ROI equalization for budget optimization with minimum spend constraints
- Interactive 6-tab React dashboard with executive briefing, scenario simulator, and AI-generated insights

**Business scope:**

- 104 weeks of weekly channel-level spend and revenue across 6 paid media channels
- Channel-specific saturation curves (k), carryover effects (decay + lag), and response coefficients (beta)
- Actionable reallocation recommendations with phased implementation guidance
- Nuanced treatment of upper-funnel channels (TV/OTT) and cross-channel halo effects

---

## Project Structure

```
marketing-mix-model/
├── model/
│   └── mmm_retail.py              # Full MMM pipeline: data gen → fit → optimize → export
├── dashboard/
│   └── marketing_budget_optimizer.jsx   # Interactive React dashboard (6 tabs)
├── data/
│   └── dashboard_data.json        # Exported model results for dashboard consumption
├── docs/
│   └── methodology.md             # Mathematical specification and design decisions
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
| Weekly budget | $175,000 |
| Total revenue | $85.2M |
| Total media spend | $19.2M |
| Overall ROAS | 4.4x |

### Approach

**Model specification:** Log-linear regression with three key transformations applied before estimation:

1. **Saturation:** `1 − exp(−spend / k)` — captures diminishing returns at channel-specific inflection points
2. **Adstock:** Geometric decay with channel-specific lag and decay rate — models carryover effects (e.g., TV/OTT: 2-week lag, 35% decay)
3. **Controls:** Trend, annual seasonality (sin/cos), and holiday indicators (Nov/Dec) to isolate media impact

**Baseline decomposition:** The baseline (revenue with zero media spend) is extracted directly from the model's control variable coefficients: `Baseline = Intercept + Trend + Seasonality + Holiday`. This is a model output, not an input assumption — following the standard approach used by Meta Robyn, Google Meridian, and PyMC-Marketing.

**Optimization:** Marginal ROI equalization with per-channel minimum spend floors. The optimizer iteratively allocates incremental budget to the channel with the highest marginal return until the total budget is exhausted.

### Key Results

| Channel | Current | Optimized | ROI | Saturation |
|---|---|---|---|---|
| Paid Search | $45,000/wk | $38,500/wk ↓ | 3.2x | 44% |
| Paid Social | $35,000/wk | $28,000/wk ↓ | 3.4x | 44% |
| Display | $20,000/wk | $12,250/wk ↓ | 2.7x | 32% |
| Email | $8,000/wk | $29,138/wk ↑ | 12.8x | 38% |
| TV / OTT | $55,000/wk | $38,500/wk ↓ | 1.5x | 35% |
| Affiliate | $12,000/wk | $28,612/wk ↑ | 7.2x | 36% |

**Projected lift:** +24.4% media-attributable revenue (~$6.8M/year) from reallocation alone.

### Important Caveats

The model has known limitations that would need to be addressed in a production setting:

- **TV/OTT undervaluation:** MMMs structurally undercount upper-funnel channels. TV drives branded search volume and direct traffic that gets attributed elsewhere. Geo-holdout tests are needed to validate.
- **Cross-channel halo effects:** Channels interact (TV → Search intent, Social → Email engagement), but the model treats them independently. Budget cuts should be phased gradually and monitored for cross-channel degradation.
- **Synthetic data:** The dataset is simulated with known ground-truth parameters. A real implementation would use actual spend/revenue data and require additional validation (out-of-sample testing, posterior predictive checks for Bayesian variants).

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
pip install numpy pandas
python model/mmm_retail.py
```

This generates `dashboard_data.json` with all model results.

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

This project mirrors the workflows of three widely-used open-source MMM frameworks:

| Framework | Organization | Approach | Best For |
|---|---|---|---|
| [Robyn](https://github.com/facebookexperimental/Robyn) | Meta | Ridge + Nevergrad | Rapid iteration, automated hyperparameter tuning |
| [Meridian](https://github.com/google/meridian) | Google | Bayesian + TFP | Geo-level data, uncertainty quantification |
| [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) | PyMC Labs | Bayesian + PyMC | Flexible custom extensions, largest community |

The custom implementation here uses the same core transforms (saturation + adstock) but keeps estimation as interpretable OLS — appropriate for the dataset size (104 weeks, 6 channels) and the project's emphasis on transparency.

---

## Author

**Freena Wang** — Senior Marketing & Product Analytics Professional

Built as a portfolio project demonstrating end-to-end marketing measurement capabilities: from econometric modeling through interactive stakeholder communication.

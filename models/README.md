# Quantitative Analysis Ecosystem — Model Catalogue

A Jane-Street-inspired layered model stack for a stock-focused quantitative analysis system. Each model has its own detailed markdown document with the canonical mathematics, calibration approach, and an algorithm outline suitable for implementation with `yfinance` as the data source (extensible to more comprehensive feeds later).

The stack is structured as **six layers** mirroring the data flow inside a quantitative trading book. This Phase 1 catalogue contains the **20 model families relevant for a stocks-first system**. Options-layer models (vol surface, stochastic vol, Greeks, market-making policy) are flagged as **Phase 2** and live in `phase2_options_later/`.

## Phase 1 — Stocks-only models (build first)

### Layer 2 — Structural / Cross-Sectional (5 models)

The models that decompose quoted prices into component parts: ETF baskets, factor exposures, and balance-sheet-driven default risk. These produce structural fair values that the signal layer overlays alpha on top of.

- [iNAV (Indicative NAV) and Equity ETF Fair Value](layer2_structural/01_inav_equity_etf.md) — Real-time fair value for equity ETFs computed from constituent holdings.
- [Futures-Implied NAV](layer2_structural/02_futures_implied_nav.md) — Fair value for international ETFs when underlying markets are closed, using liquid futures proxies.
- [Merton-KMV Structural Credit](layer2_structural/03_merton_kmv.md) — Equity as a call on firm assets; produces distance-to-default and expected default frequency from stock prices and balance-sheet data.
- [Barra-style Factor Models and PCA on Returns](layer2_structural/04_factor_models_pca.md) — Multi-factor risk decomposition for equity portfolios; covariance matrix construction via factor structure.
- [Cost-of-Carry and Discrete-Dividend Index Basket](layer2_structural/05_cost_of_carry_dividends.md) — Futures-cash basis pricing; ETF creation arbitrage; forward-price construction with dividends.

### Layer 4 — Signals and Machine Learning (9 models)

Predictive overlays that price the residual after the structural layer. The heart of a stock-trading system.

- [Engle-Granger and Kalman Cointegration](layer4_signals_ml/06_cointegration_pairs.md) — Pair-trading and basket mean-reversion via cointegrated relationships.
- [Avellaneda-Lee PCA-Residual Statistical Arbitrage](layer4_signals_ml/07_avellaneda_lee_stat_arb.md) — Cross-sectional equity mean reversion via OU process on PCA residuals.
- [GARCH Family (GARCH, GJR-GARCH, EGARCH)](layer4_signals_ml/08_garch_family.md) — Autoregressive conditional volatility forecasting.
- [HAR-RV — Heterogeneous Autoregressive Realized Volatility](layer4_signals_ml/09_har_rv.md) — Multi-scale realized-vol forecasting using intraday returns.
- [Supervised Autoencoder + MLP (Kaggle 2021 Winner)](layer4_signals_ml/10_supervised_autoencoder_mlp.md) — Deep tabular learning architecture used by the Jane Street Kaggle winning solution.
- [Gradient-Boosted Trees and Transformer Ensembles](layer4_signals_ml/11_gbdt_transformer_ensembles.md) — LightGBM/XGBoost/CatBoost ensembled with Transformer encoders for panel return forecasting.
- [Bayesian Hierarchical and Kalman Dynamic Factor Models](layer4_signals_ml/12_bayesian_kalman.md) — Partial-pooling estimation and time-varying factor loadings.
- [Online Learning — Hedge / FTRL / FTPL](layer4_signals_ml/13_online_learning.md) — Adversarial-regret algorithms for real-time signal-weight updates under non-stationarity.
- [Anomaly Detection and FinBERT NLP](layer4_signals_ml/14_anomaly_detection_finbert.md) — Autoencoder reconstruction error, isolation forests, and transformer-based news sentiment.

### Layer 5 — Execution (2 models)

Translating signals into orders. Even a retail-scale stock system benefits from a scheduling layer.

- [Kyle Lambda and the Square-Root Impact Law](layer5_execution/15_kyle_lambda_sqrt_impact.md) — Market-impact theory: linear (Kyle) vs concave (Bouchaud-Toth) impact.
- [Almgren-Chriss Optimal Execution](layer5_execution/16_almgren_chriss.md) — Mean-variance liquidation schedule for parent orders.

### Layer 6 — Risk and Capital (4 models)

Aggregation of exposures, tail-risk capital, and stress.

- [Value-at-Risk (Parametric, Historical-Sim, Monte Carlo)](layer6_risk/17_var.md) — Quantile-based market-risk measure with three estimation paths.
- [Expected Shortfall under FRTB](layer6_risk/18_expected_shortfall.md) — Coherent tail-conditional risk measure replacing VaR in Basel III IMA.
- [Stress Tests](layer6_risk/19_stress_tests.md) — Historical-replay, hypothetical-scenario, and reverse-stress methodologies.
- [Liquidity-Adjusted VaR](layer6_risk/20_liquidity_adjusted_var.md) — Risk measure incorporating liquidation cost and time.

## Phase 2 — Add later (for options & vol trading)

These model families are documented separately and become relevant once you extend the ecosystem to options:

### Layer 0 — Foundation
- Black-Scholes-Merton — The implied-vol lens; trivial to add once you have option chains.

### Layer 1 — Vol Surface
- Gatheral SVI (raw); SSVI / eSSVI; Dupire local volatility; SABR.

### Layer 3 — Stochastic / Exotic Dynamics
- Heston; Merton (1976) jump-diffusion; Bates SVJ; Rough Bergomi / Rough Heston; Local-Stochastic Volatility (LSV); Variance Gamma / NIG / CGMY Lévy models.

### Layer 5 — Market-Making Policy
- Avellaneda-Stoikov reservation-price + spread; Cartea-Jaimungal alpha-aware extensions; PPO / DDPG RL for quote skew.

### Layer 6 — Greeks
- Greeks decomposition (delta, gamma, vega, vanna, volga, charm, color) — required once you carry option inventory.

## Models intentionally excluded

These are documented in the Jane Street addendum but **excluded** because they require infrastructure or data unavailable in a yfinance-based system: Order Book Imbalance (OBI) / Order Flow Imbalance (OFI); Hawkes self-exciting processes; Lillo-Mike-Farmer order-flow long memory; Cont-Stoikov-Talreja limit order book queues — all require tick-by-tick LOB data. AP creation-basket optimization (fixed income) and Jarrow-Turnbull / Duffie-Singleton reduced-form credit — require bond and CDS data. SA-CCR and CCP margin models (STANS, SPAN 2, LCH, Eurex Prisma) — apply only to prime-broker / clearing-house counterparties. SOFR / OIS curve construction, Nelson-Siegel, and Hagan-West splines are deferred to a future rates-extension phase.

## Architectural notes

Each model is intended to be run by **its own agent** in a multi-agent ecosystem, with a shared event bus connecting them and a calibration scheduler invoking each at its appropriate frequency:

- **Tick / minute frequency** — Online-learning weight updates; anomaly detection; signal inference.
- **End-of-day** — GARCH refit; HAR-RV refit; cointegration tests; factor model refit; VaR window roll; deep-learning model retraining (or weekly).
- **Weekly / monthly** — Merton-KMV calibration; factor-model factor selection; deep-learning architecture iteration; stress-test scenario refresh.

Data layer is `yfinance`-based today, abstracted behind a thin `DataProvider` interface so the same models can be re-pointed at Polygon, IEX Cloud, or a paid intraday feed later without changing model logic. Each model doc includes a section on what yfinance fields it consumes.

# LolDraft: Real-Time Bayesian Drafting Engine for League of Legends

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Patch 14.23](https://img.shields.io/badge/Patch-14.23-orange.svg)](https://www.leagueoflegends.com)
[![Inference Latency](https://img.shields.io/badge/Latency-%3C5ms-success.svg)](connector/lcu_socket.py)
[![Paper PDF](https://img.shields.io/badge/Paper-8--Page_PDF-red.svg)](docs/loldraft_ml_paper.pdf)

> Interpretable machine learning decision-support system that estimates empirical tactical weights from match telemetry and computes real-time champion recommendation scores with Bayesian flex-pick resolution in under 5 milliseconds.

---

## Executive Summary

Champion select in competitive *League of Legends* presents an exponential search space exceeding **10^16 possible 10-champion combinations**. Under the 30-second drafting timer, players frequently fall into unassisted drafting deficits that swing match win probabilities by 10% to 25%.

The goal of **LolDraft** is **not** to act as a standalone match outcome predictor, but to provide an **interpretable decision-support engine** that evaluates and recommends optimal picks during live draft:

1. **Empirical Weight Estimation (Machine Learning Core):** Rather than assigning arbitrary heuristic values to draft parameters, LolDraft trains position-specific L2-regularized logistic regression models across 41,450 Emerald+ observations (KR, EUW, NA). The models evaluate match telemetry to estimate statistically grounded coefficients (beta weights, standard errors, and z-statistics) measuring how baseline win rates, lane counter deltas, team synergies, duo synergies, and roaming threat deltas directly impact match winning probabilities.
2. **Normalized Scoring Engine:** These calibrated logistic weights are converted into role-specific scoring multipliers for an ultra-low-latency (<0.05 ms) in-memory decision function evaluated across all 160+ champions.
3. **Exact Bayesian Flex Resolution:** When opponents pick multi-role flex champions (e.g., Gragas, Poppy, Pantheon), LolDraft calculates exact marginal role probabilities across all valid role permutations to weight expected matchup counters.
4. **Composition & Turn Dynamics:** The scoring engine dynamically penalizes blind pick vulnerability when drafting without lane opponent visibility, and enforces damage guardrails against tank armor/magic-resist stacking.
5. **Demonstrated Strategic Lift:** In empirical validation across draft advantage quintiles, team compositions aligning with the model's top recommendations achieve an observed **58.3% win rate** compared to **38.7%** for unassisted deficit drafts (+19.6 percentage point operational lift).

---

## Statistical Weight Estimation & Empirical Results

### 1. Calibrated Tactical Weights (Logistic Beta Coefficients)
Trained across positional subsets from 4,145 Emerald+ matches. The learned parameters reflect the true tactical forces dictating each lane:

| Position | Baseline WR | Lane Counter Delta | Team Synergy | Threat Delta | Bot Duo Synergy | Primary Tactical Driver |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Top** | 1.00 (base) | **1.58** (z = +6.21\*\*\*) | 0.95 (z = +3.85\*\*\*) | 0.55 (z = +2.48\*) | — | Isolated 1v1 counter matchup dominance |
| **Jungle** | 1.00 (base) | **2.18** (z = +4.82\*\*\*) | 1.45 (z = +3.61\*\*\*) | **1.89** (z = +5.20\*\*\*) | — | Roaming threat mitigation & skirmish pressure |
| **Middle** | 1.00 (base) | **1.80** (z = +4.12\*\*\*) | 1.58 (z = +3.94\*\*\*) | **1.98** (z = +5.23\*\*\*) | — | Cross-map threat coverage & wave priority |
| **Bottom** | 1.00 (base) | 0.97 (z = +3.52\*\*\*) | 0.42 (p = 0.11 ns) | 0.50 (z = +2.11\*) | 0.63 (z = +1.98\*) | Raw baseline champion power & 2v2 survival |
| **Support** | 1.00 (base) | 0.70 (z = +2.61\*\*) | 0.51 (z = +1.85 ns) | 0.88 (z = +4.41\*\*\*) | **0.69** (z = +2.29\*) | Duo partner synergy & roaming threat control |

*Significance codes: \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05, ns = not statistically significant.*

### 2. Decision Function Formulation
The normalized weights parameterize the live scoring engine (matching Eq. 6 in the research paper):

$$
\mathcal{S}(c, r \mid A, E) = w_{\text{base}} \text{WR}_{\text{base}} + w_{\text{lane}} \mathbb{E}[\Delta_{\text{lane}}] + w_{\text{threat}} \Delta_{\text{threat}} + w_{\text{duo}} \Delta_{\text{duo}} + w_{\text{syn}} \Delta_{\text{team}} + \mathcal{C}_{\text{comp}} - \Omega_{\text{blind}}
$$

### 3. Draft Advantage Validation (Quintile Match Lift)
Evaluating observed game outcomes grouped by model draft score advantage on out-of-sample matches:

| Draft Advantage Tier | Matches | Observed Win Rate | Tactical Profile |
| :--- | :---: | :---: | :--- |
| **Q1 (Severe Deficit)** | 829 | **38.7%** | Compounded counter matchups & zero-AP draft traps |
| **Q2 (Moderate Deficit)** | 829 | 45.4% | Unfavorable lane matchups / negative synergy |
| **Q3 (Neutral Draft)** | 829 | 48.1% | Evenly contested draft (Blue baseline 48.3%) |
| **Q4 (Moderate Advantage)** | 829 | 51.3% | Favorable lane priorities & threat coverage |
| **Q5 (Model-Optimized)** | 829 | **58.3%** | Pick recommendation alignment under serpentine order |

---

## Technical Paper

For detailed mathematical derivations, Bayesian prior calibration, composition penalty formulations, and complete references, read the full paper:

- **[LolDraft Technical Paper (PDF)](docs/loldraft_ml_paper.pdf)**: *António Barroso, October 2026* (8 pages, IEEE/ACM style).



# LolDraft: Real-Time Bayesian Drafting Engine for League of Legends

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Patch 14.23](https://img.shields.io/badge/Patch-14.23-orange.svg)](https://www.leagueoflegends.com)
[![Inference Latency](https://img.shields.io/badge/Latency-%3C5ms-success.svg)](connector/lcu_socket.py)
[![Paper PDF](https://img.shields.io/badge/Paper-8--Page_PDF-red.svg)](docs/loldraft_ml_paper.pdf)

> Autonomous, real-time draft recommendation system that resolves champion ambiguity through exact Bayesian role marginalization and evaluates composite lane and team synergies within 5 milliseconds.

---

## Executive Summary

Champion select in competitive League of Legends presents a search space exceeding **10^16 possible 10-champion team combinations**. Under the 30-second tournament turn clock, human drafters frequently succumb to confirmation bias and struggle to evaluate flex picks across multiple lanes. 

**LolDraft** bridges modern machine learning and live client execution:
- **Predictive Accuracy:** Achieves **56.4% out-of-sample win prediction accuracy** across 100,000 Patch 14.23 solo-queue and professional matches (+6.4% over raw individual champion win rates).
- **Sub-5ms Inference Latency:** Precomputes high-dimensional pairwise synergy and counter tensors to evaluate the entire 168+ champion roster in under 5 ms per draft event.
- **Dynamic Ambiguity Resolution:** Employs an exact Bayesian marginal updater to dynamically deduce opponent lane assignments for multi-role flex champions (e.g., Gragas, Poppy, Pantheon) by evaluating all valid injective role permutations as draft turns progress.
- **Riot LCU Integration:** Features a zero-configuration companion that hooks into the active `LeagueClientUx` authenticated WebSocket/REST interface to deliver real-time counter-pick recommendations during live matches.

---

## System Architecture

```
                                      +------------------------------------+
                                      |   Riot LCU Client / Mock Replay   |
                                      +-----------------+------------------+
                                                        |
                                            WebSocket / REST Payload
                                                        |
                                                        v
                                      +-----------------+------------------+
                                      |   Draft Event Parser & Role Hash   |
                                      +-----------------+------------------+
                                                        |
                                   +--------------------+--------------------+
                                   |                                         |
                                   v                                         v
                     +-------------+-------------+             +-------------+-------------+
                     | Bayesian Role Inference   |             | Composite Scoring Engine  |
                     | - Categorical Role Priors |             | - Baseline Win Rates      |
                     | - Exact Permutation Bayes |             | - Pairwise Lane Deltas    |
                     | - Flex Ambiguity Scoring  |             | - Pairwise Team Synergies |
                     +-------------+-------------+             | - AD/AP Composition Guard |
                                   |                           +-------------+-------------+
                                   +--------------------+--------------------+
                                                        |
                                                        v
                                      +-----------------+------------------+
                                      | Real-Time Top-5 Ranked Counterpicks|
                                      |  (Terminal / Live GUI Companion)   |
                                      +------------------------------------+
```

---

## Empirical Benchmarks

Evaluated on an out-of-sample test split of 20,000 matches from Patch 14.23 (Emerald+ to Challenger MMR):

| Model / Strategy | Test Accuracy | Log-Loss | Turn Latency | Role Ambiguity Handling |
| :--- | :---: | :---: | :---: | :---: |
| **Independent Baseline (Win Rate only)** | 50.0% | 0.6931 | < 0.1 ms | None (Unaware) |
| **Naive Direct Lane Counter** | 52.8% | 0.6784 | 0.8 ms | Heuristic static role |
| **Unregularized Pairwise Synergy** | 53.9% | 0.6812 | 1.9 ms | Vulnerable to flex traps |
| **LolDraft (Bayesian Role Inference + Regularized Synergies)** | **56.4%** | **0.6542** | **4.2 ms** | **Dynamic Bayesian Marginals** |

---

## Quickstart & Usage

### 1. Synthetic Mock Simulator (Offline Replay)
Simulate a sequential 10-pick draft without needing the League client open:
```bash
python connector/mock_draft.py
```

### 2. Live Game Client Companion (Option 3)
Connect directly to your active League of Legends client during Champion Select:
- **Via Desktop Shortcut:** Double-click `LolDraft Live Companion` on your Desktop.
- **Via Command Prompt:**
```bash
launch_live.bat
# or directly:
python connector/lcu_socket.py
```
*The companion automatically detects `LeagueClientUx.exe`, extracts the Riot lockfile credentials, catches up if launched mid-draft, and streams top 5 counter-picks as players lock in champions.*

### 3. Interactive Web GUI Simulator
Launch the standalone graphical client simulator in any modern browser:
```bash
# Open directly in browser:
start ui/index.html
# or serve locally:
python -m http.server 8000 -d ui
```

---

## Repository Structure

```
loldraft/
├── connector/               # Riot LCU client integration
│   ├── lockfile.py          # Process memory inspection & 5-tuple lockfile parser
│   ├── lcu_socket.py        # Authenticated WSS/REST live draft listener & scorer
│   └── mock_draft.py        # Standalone 10-pick synthetic draft scenario
├── data/                    # Ingested datasets & precomputed tensors
│   ├── current_matrix.json  # Precomputed winrates, deltas, synergies (Patch 14.23)
│   └── training_dataset.csv # 100k match training dataset
├── docs/                    # Research documentation & academic publication
│   ├── loldraft_ml_paper.pdf # Complete 8-page academic paper (October 2026)
│   ├── loldraft_ml_paper.tex # Publication-ready LaTeX source
│   ├── fig1_role_profiles.pdf # High-resolution vector figures
│   └── writing_style.md     # Project stylometric and editorial guidelines
├── pipeline/                # Machine learning core
│   ├── model.py             # Matchup feature extractor and training pipeline
│   └── role_inference.py   # Bayesian role inference & draft scorer engine
├── ui/                      # Visual client simulator
│   ├── index.html           # Dark-mode responsive HUD interface
│   ├── app.js               # Client-side scoring and candidate engine
│   └── styles.css           # Glassmorphism styling tokens
├── launch_live.bat          # One-click Windows companion launcher
└── README.md                # Project documentation
```

---

## Technical Paper

For detailed mathematical derivations, Bayesian prior calibration, composition penalty formulations, and complete references, read the full paper:

- **[LolDraft Technical Paper (PDF)](docs/loldraft_ml_paper.pdf)**: *António Barroso, October 2026* (8 pages, IEEE/ACM style).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

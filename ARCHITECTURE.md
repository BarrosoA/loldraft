# LolDraft: System Architecture Specification

## 1. Executive Overview

LolDraft is an ultra-low-latency real-time drafting companion for League of Legends. It connects directly to the local game client, tracks champion select events via WebSocket, and computes explainable, mathematically calibrated champion recommendations in under 5 milliseconds.

Instead of unreliable black-box deep learning or naive flat win-rate tables, LolDraft uses **Pairwise Matrix Factorization with Regularized Logistic Regression Calibration**. This decouples heavy statistical processing from live drafting, ensuring maximum statistical reliability, full explainability, and near-instantaneous execution during the 30-second pick timer.

---

## 2. High-Level System Architecture

The system is strictly divided into two distinct execution domains:

```
[ Domain 1: Offline / Patch Pipeline (Runs once per 14-day patch) ]
   Riot Match-V5 API / Aggregated Feeds
                  │
                  ▼
   1. Data Ingestion & Sanitization
                  │
                  ▼
   2. Pairwise Matrix Computation & Empirical Bayes Smoothing
                  │
                  ▼
   3. Logistic Regression Calibration (Optimal Feature Weights)
                  │
                  ▼
          [ loldraft.db / current_matrix.json ] (< 25 MB)

──────────────────────────────────────────────────────────────────

[ Domain 2: Real-Time Companion (Runs locally during Champ Select) ]
   League Client (LCU) WebSocket (Port, Password from lockfile)
                  │
                  ▼
   1. Event Filter & Draft State Manager (Filters non-action noise)
                  │
                  ▼
   2. Turn Context Classifier (First Pick Blind Safety vs Late Counter)
                  │
                  ▼
   3. In-Memory Recommendation Engine (< 5ms scoring loop)
                  │
                  ▼
   4. Compositional Guardrails (Damage profiles, CC presence)
                  │
                  ▼
   5. User Interface Overlay (Top picks, deltas, rationale badges)
```

---

## 3. Data Ingestion & Offline Pipeline (Domain 1)

### 3.1. Raw Data Extraction
* **Target Population:** High-ELO (Emerald+) ranked solo queue match data for the current active patch.
* **Extraction Modalities:**
  * *Option A (Direct Crawler):* Graph-traversal via Riot API Match-V5 (`GET /lol/match/v5/matches/{matchId}`). Requires ~52,000 requests for 50,000 matches.
  * *Option B (Aggregated Importer - Recommended for MVP):* Ingest pre-aggregated patch statistical tables (~170 champion JSON payloads) in under 30 seconds.
* **Sample Size Threshold:** Minimum 50,000 matches per patch to ensure pairwise statistical significance.

### 3.2. Matrix Generation
For all ~168 champions and 5 standard roles (Top, Jungle, Mid, Bot, Support), four core statistical matrices are computed:

1. **Role Baseline Matrix [B]:**
   Prior win rate for champion c in role r:
   Baseline(c, r) = Wins(c, r) / TotalGames(c, r)

2. **Direct Lane Counter Matrix [C_lane]:**
   Head-to-head delta for champion c against enemy e in the same role r:
   Delta_Lane(c, e, r) = WinRate(c vs e in r) - Baseline(c, r)

3. **Ally Synergy Matrix [S_ally]:**
   Co-occurrence delta for champion c with ally a:
   Delta_Synergy(c, a) = WinRate(c with a) - Baseline(c, r)

4. **Enemy Threat Matrix [T_threat]:**
   Cross-map counter delta for champion c against enemy e in a different role:
   Delta_Threat(c, e) = WinRate(c vs e across match) - Baseline(c, r)

5. **Blind Pick Vulnerability Index [V_blind]:**
   A metric representing how severely champion c is punished when counter-picked blindly:
   V_blind(c, r) = Average negative Delta_Lane against the top 5 most common counter-picks to c.

### 3.3. Statistical Smoothing (Empirical Bayes Shrinkage)
To prevent low-sample anomalies (e.g., an off-meta pick with 3 wins out of 4 games showing a +25% delta), all deltas are shrunk towards zero using empirical Bayes smoothing:
   Smoothed_Delta = (Raw_Delta * Sample_Size) / (Sample_Size + M)
Where M is the smoothing constant (default: M = 200 games). When sample size is small, Smoothed_Delta approaches 0.0 (neutral impact).

### 3.4. Weight Calibration (Logistic Regression)
Rather than guessing arbitrary weights, a Ridge Logistic Regression model is trained once per patch across 50,000 matches:
* **Features (X):**
  * x1 = Baseline(c, r)
  * x2 = Delta_Lane(c, enemy_laner, r)
  * x3 = Sum of Delta_Synergy(c, locked_allies)
  * x4 = Sum of Delta_Threat(c, locked_enemies)
* **Target (y):** Match Outcome (Win = 1, Loss = 0).
* **Outputs:** Calibrated coefficients (w_base, w_lane, w_synergy, w_threat).

---

## 4. Real-Time Engine (Domain 2)

### 4.1. Local Client (LCU) Integration
* **Process Detection:** Reads `lockfile` from the active League of Legends directory:
  Format: `LeagueClient:PID:Port:AuthToken:Protocol`
* **Transport:** Connects via authenticated WSS (WebSocket Secure) to:
  `wss://127.0.0.1:<Port>/`
  *Authentication:* HTTP Basic Auth with username `riot` and password `<AuthToken>`.
  *TLS:* Must bypass self-signed certificate validation (`rejectUnauthorized: false` / `ssl_verify=False`).
* **Subscribed Event:** `OnJsonApiEvent_lol-champ-select_v1_session`
* **Event Filtering Optimization:**
  LCU emits events on timer ticks, trades, and chat. The event handler must filter out non-action updates and only trigger scoring re-evaluations when:
  1. An action transitions to `completed = true` (champion locked or banned).
  2. The active turn shifts to the local player.

### 4.2. Turn Context & Scoring Formulation
The recommendation formula automatically adapts based on whether the player is **First/Blind Picking** or **Counter-Picking**:

#### Scenario A: Later Pick (Enemy Laner Already Locked)
Full counter exploitation formula:
   Final_Score(c) = w_base * Baseline(c)
                  + w_lane * Delta_Lane(c, enemy_laner)
                  + w_synergy * Sum(Delta_Synergy(c, locked_allies))
                  + w_threat * Sum(Delta_Threat(c, locked_enemies))
                  + Composition_Penalty(c)

#### Scenario B: First / Early Pick (Enemy Laner Unknown)
When picking blindly into an unrevealed opponent, `Delta_Lane` is unknown. Relying solely on baseline win rate is flawed because volatile champions (e.g., Kayle, Kassadin) get brutally counter-picked. The engine incorporates the Blind Vulnerability Index:
   Final_Score(c) = w_base * Baseline(c)
                  - w_blind * V_blind(c)
                  + w_synergy * Sum(Delta_Synergy(c, locked_allies))
                  + w_threat * Sum(Delta_Threat(c, locked_enemies))
                  + Composition_Penalty(c)

### 4.3. Compositional Guardrails & Multipliers
Statistical deltas cannot always identify glaring strategic voids. The engine applies explicit composition rules:
1. **Damage Profile Imbalance:**
   * If the locked team is >= 80% Physical Damage, apply -3.0% penalty to pure AD candidates and +2.5% bonus to primary AP candidates.
   * If the locked team is >= 80% Magic Damage, apply inverse penalties.
2. **Crowd Control & Frontline Deficit:**
   * If 0 hard CC or 0 frontline champions are locked across 4 picks, apply +2.0% bonus to heavy CC/engage champions and -2.0% penalty to squishy assassins.

### 4.4. Ambiguous Flex Pick Resolution
When an enemy locks a flex champion (e.g., Swain, Yasuo, Gragas) before lane assignments are clear:
* The engine uses historical role-frequency priors:
  P(Role = Mid), P(Role = Support), P(Role = Top).
* Lane delta is computed as an expectation across possible roles:
  Expected_Lane_Delta = Sum over roles [ P(Role) * Delta_Lane(c, enemy, Role) ]

---

## 5. Storage Schema & Performance Specifications

### 5.1. Lookup Matrix Schema (`current_matrix.json` / SQLite)
```json
{
  "patch": "14.18",
  "weights": {
    "base": 1.0,
    "lane": 1.38,
    "synergy": 0.85,
    "threat": 0.62,
    "blind": 0.90
  },
  "champions": {
    "236": {
      "name": "Lucian",
      "roles": {
        "bottom": {
          "baseline": 50.8,
          "blind_vulnerability": 1.8,
          "counters": { "51": -2.4, "222": +1.9 },
          "synergies": { "412": +2.8, "201": +3.2 },
          "threats": { "157": -1.5, "105": -2.1 }
        }
      }
    }
  }
}
```

### 5.2. Performance Benchmarks
| Metric | Target Specification |
| :--- | :--- |
| **Matrix Memory Footprint** | < 25 MB RAM (entire matrix preloaded in memory) |
| **Engine Evaluation Latency** | < 5 ms for full 168-champion ranking |
| **LCU WebSocket Response** | < 10 ms from client pick to UI refresh |
| **Database Disk Footprint** | < 30 MB (compressed JSON or SQLite) |
| **Cold Start Boot Time** | < 300 ms |

---

## 6. Project Directory Structure

```
loldraft/
├── ARCHITECTURE.md                  # System architecture specification
├── pyproject.toml / package.json    # Project configuration and dependencies
├── data/
│   ├── patches/                     # Historical patch archives
│   └── current_matrix.json          # Active patch lookup tables & calibrated weights
├── pipeline/                        # Domain 1: Offline ETL & Calibration
│   ├── __init__.py
│   ├── ingestion.py                 # Fetcher for match logs or bulk tables
│   ├── matrix_builder.py            # Pairwise delta computation & Bayes smoothing
│   └── calibrator.py                # Logistic regression weight estimation
├── core/                            # Domain 2: Live Scoring Engine
│   ├── __init__.py
│   ├── state_manager.py             # Draft state parsing & flex resolver
│   ├── scoring_engine.py            # Vectorized formula evaluation (< 5ms)
│   └── heuristics.py                # Damage profile and CC guardrails
├── connector/                       # Domain 2: Local Client Hook
│   ├── __init__.py
│   ├── lockfile.py                  # LCU process detection & credentials
│   └── lcu_socket.py                # WebSocket client for real-time champ select
└── ui/                              # Presentation Layer
    ├── index.html
    ├── overlay.css
    └── app.js                       # Lightweight HUD overlay
```

---

## 7. Verification & Testing Strategy

1. **Synthetic Draft Stress Testing:**
   * Script a mock WebSocket draft sequence (Bans -> Pick 1 to 10) feeding into `scoring_engine.py`.
   * Assert execution time remains strictly under 5ms per event.
2. **Empirical Validation Against Baseline:**
   * Backtest the model against 10,000 historical matches: calculate predicted win rate vs actual match outcomes.
   * Verify Brier score and log-loss improvement over static tier-list predictions.
3. **Graceful Degradation:**
   * Ensure that when the user is First Pick (zero enemy picks, zero ally picks), the engine cleanly applies `Baseline - Blind_Vulnerability` without throwing null errors.

# Author Writing Style Profile: António Barroso
### Universal Applied Machine Learning & Data Science Report Specification

*Derived from stylometric and linguistic analysis of António Barroso's research papers (Erasmus School of Economics, 2021) and extended with production-grade ML engineering patterns (LolDraft, 2026).*

---

## RULE 0: PROJECT TRUTH PRECEDENCE (Anti-Hallucination Mandate)

> [!CRITICAL]
> **THIS SPECIFICATION GOVERNS VOICE, RHETORIC, SYNTAX, AND FORMATTING DISCIPLINE — NEVER PROJECT FACTS.**
>
> When an AI agent applies this style guide to a new project:
> 1. **Zero Hallucinated Frameworks:** You must extract 100% of the problem context, dataset volumes, model types, feature names, hyperparameter values, and performance metrics **directly from the target project's active codebase and verified execution outputs**.
> 2. **Never Force a Mismatched Comparative Framing:**
>    - If the project trained **one calibrated model architecture** (e.g., a multi-position regularized logistic regression, a single neural network, or a pipeline with sub-domain models), report that single architecture honestly across its roles, feature subsets, or operational cohorts. **Never invent artificial strawman models** (e.g., "Model 1 vs. Model 2" or "Naive Bayes Baseline") merely because a reference paper compared multiple models.
>    - If the project actually trained and benchmarked multiple algorithms (e.g., Random Forest vs. XGBoost vs. LightGBM), compare those real models using the comparative templates in Section 5.
> 3. **Never Fabricate Scale or Mechanism:** If the target project does not involve combinatorics, do not mention "quadrillions." If it does not use Empirical Bayes shrinkage, do not include shrinkage equations. Describe the *actual* engineering solutions, data preprocessing steps, and operational constraints implemented in that repository.
> 4. **Violation Penalty:** Any agent that fabricates a baseline model, invents a mathematical derivation not grounded in the code, or inserts fictional results to satisfy a stylistic template has committed a fatal hallucination error.

---

## 1. The Linguistic DNA of António Barroso

### A. Perspective and Voice
* **Perspective:** Active first-person plural (`we decided`, `we suspected`, `we opted`, `our model`, `our solution`). Passive voice is strictly minimized (prefer *"We cleaned match instances where..."* over *"Instances were cleaned by the authors"*).
* **Persona:** Applied data scientist and quantitative researcher explaining modeling, statistical calibration, and architectural tradeoffs to technical peers. Direct, grounded, and devoid of marketing fluff.
* **Tone:** Pragmatic, objective, and modest. Claims are evidence-based and understated (`"This report presented a modest application..."`, `"we suspect this is just a side effect of reducing overfitting"`, `"shifts point estimates by less than 0.028"`).

### B. Sentence Mechanics and Cadence
* **Sentence Length:** Crisp and purposeful. The average sentence length is **17 words**, avoiding the sprawling multi-clause compound sentences common in generic AI writing.
* **Causal Sequencing:** Paragraphs follow a strict cause-and-effect narrative:
  $$\text{Problem / Observation} \longrightarrow \text{Hypothesis / Risk} \longrightarrow \text{Our Solution} \longrightarrow \text{Empirical Impact}$$
* **Sentence-Initial Relative Clauses:** A signature stylistic marker is using short, sentence-initial relative fragments for rhetorical emphasis:
  - *"Which is the main goal of the model."*
  - *"Which represents a significant cost-saving benefit."*

### C. Cognitive Framing & Stylistic Habits
* **Engineering Utility First:** Every modeling decision (loss function, regularization, feature transform, threshold pruning) is defended by a concrete operational objective: reducing sample variance, eliminating multicollinearity, preventing rank flips, halving training runtime, or meeting live millisecond latency budgets.
* **Latinate Cognitive Constructions:** The writing features natural Portuguese/Romance-influenced English phrasings that give it an authentic, scholarly cadence:
  - *"In this section, we ought to [compare/evaluate]..."* (standard transition marker)
  - *"We incur in the risk of [failing to generalize / overfitting]..."*
  - *"Amid of the procedures described above..."*
  - *"Due to the [imbalanced / high-dimensional] nature of our classification problem..."*
  - *"We concluded, based on the posterior results, that..."*
  - *"Setting these parameters offers an opportunity to consider the trade-off between [A] and [B], like [tangible resource]..."*

### D. The "AI Slop" Ban List (Strictly Prohibited Words)
Generic LLMs default to dramatic, decorative prose. The following words and clichés are **strictly banned**:

| Category | Banned AI Words / Clichés |
| :--- | :--- |
| **Decorative Verbs** | *delve, unravel, foster, showcase, highlight, underscore, shed light on, pave the way* |
| **Grandiose Adjectives** | *pivotal, paramount, testament to, beacon, tapestry, multifaceted, game-changing, groundbreaking, quintessential* |
| **Hedging Slop** | *it is worth noting that, it is important to remember, needless to say, in conclusion one might argue* |
| **Vague Adverbs** | *seamlessly, crucially, exponentially (unless mathematically derived), vastly* |

### E. The Authentic António Barroso Lexicon (Whitelist)
Use these authentic verbs, nouns, and transitional adverbs to reproduce the natural author voice:

* **Action Verbs:** *opt, suspect, discern, surmount, curb, track, monitor, alter, derive, assess, incur, tackle, examine, decouple, calibrate, prune.*
* **Nouns / Concept Tags:** *modest application, trade-off, resource consumption, real-world applicability, pattern diverseness, posterior results, operational outlook, cost-saving benefit, sweet spot, deadband.*
* **Transitional Adverbs:** *Hence, Thus, Particularly, Specifically, As a result, Consequently, Furthermore, Conversely, Amid.*

### F. The Dual-Audience Principle (Executive Clarity + Specialist Rigor)
Reports are read by both **business decision-makers** (GMs, directors, product managers) and **technical specialists** (data scientists, engineers). The prose must never alienate either audience:

1. **Intuition Before Formalism:** Always state the operational intuition in plain English *before* introducing mathematical formulas or statistical metrics.
   - *Bad (Specialist Only):* *"Under asymptotic M-estimation theory, standard errors decay as $1/\sqrt{N}$."*
   - *Good (Dual-Audience):* *"From a data engineering standpoint, estimation uncertainty diminishes with the square root of sample size ($1/\sqrt{N}$). Beyond 4,000 matches, parameter estimates stabilize, shifting by less than 0.028 points..."*
2. **Translate Statistical Metrics to Operational Impact:**
   - Specialists care about ROC-AUC, Brier score, and log-loss.
   - Executives care about **real-world win rate swings**, **hours of API quota saved**, and **millisecond client responsiveness**. Always report both.
3. **Eliminate Gratuitous Jargon & Domain Bleed:**
   - Never copy technical terms from previous projects that don't apply to the current domain (e.g., calling champion select interactions "spatio-temporal" because a previous project analyzed maritime GPS data).
   - Use direct, clean phrasing:

| Overly Dense Academic Jargon | Dual-Audience Phrasing (Business + Specialist) |
| :--- | :--- |
| *"Zero-centered residual deltas between consecutive draft interactions"* | *"Net matchup advantage isolating how much a choice outperforms its baseline"* |
| *"Bayesian role marginalization across all permutations"* | *"Expected advantage weighted by how frequently a champion appears in each role (Eq. 1)"* |
| *"Asymptotic covariance matrix reveals the forces"* | *"Evaluating learned weights and statistical significance reveals the decisive tactical forces"* |
| *"High sample variance producing deceptive win rates"* | *"Deceptive win rates driven by small-sample luck rather than true strength"* |
| *"Model-optimized compositions in Q5"* | *"Compositions aligning with model recommendations under standard serpentine pick order"* |

---

## 2. "AI-Speak" vs. "António Barroso Voice" (Side-by-Side)

| Scenario | Generic AI Prose (BANNED) | António Barroso Voice (REQUIRED) |
| :--- | :--- | :--- |
| **Data Cleaning** | *"Data preprocessing is a pivotal and multifaceted cornerstone of any robust machine learning pipeline. In this work, we meticulously delved into the intricacies of raw telemetry..."* | *"Arranging and cleaning [X] data is not a simple task. Hence, this report provides an extensive outlook on the data preparation part of our model. In order to get the most out of [X] data, we need to overcome some of its limitations..."* |
| **Feature Transformation** | *"To unlock the transformative potential of geographic features and foster superior generalizability, we ingeniously transformed the coordinates into relative vectors."* | *"We suspected that absolute values for [feature] would lead to overfitting. We incur in the risk of training our model to make predictions only for [subset], completely failing to generalize. Thus, this feature needs to be transformed for real-world applicability."* |
| **Hyperparameter Tradeoffs** | *"Fine-tuning hyperparameters serves as a powerful testament to the delicate balance between computational overhead and supreme algorithmic prowess."* | *"Setting these parameters offers an opportunity to consider the trade-off between robustness of the model and resource consumption, like time spent training."* |
| **Discussing Model Results** | *"Our groundbreaking architecture seamlessly outperforms the competition, showcasing remarkable predictive efficacy across all conceivable evaluation metrics."* | *"Our model includes all feature transformations. It performs far better than the initial configuration... As seen by previous results, we can assess that our feature selection increased model performance while cutting training time in half. Which represents a significant cost-saving benefit."* |
| **Operational Lift** | *"The model acts as a game-changer, driving unprecedented synergy and delivering monumental value to end-users."* | *"To measure operational utility in practice, we evaluated outcomes across quintiles of model advantage (Table 6). In unassisted cases, users frequently fall into severe traps... In contrast, when aligned with model recommendations, success rate surges by [N] percentage points."* |
| **Conclusion** | *"In conclusion, this study has delved into the profound landscape of machine learning, paving the way for exciting future paradigms in the field."* | *"Machine learning makes it possible to develop effective tools to tackle [domain] issues. This report presented a modest application of machine learning methods to face the challenges of [problem]. For future work we may consider expanding [X] or using neural networks to analyze [Y]."* |

---

## 3. Generalized Document Architecture: The Strict 8-Page Blueprint

When drafting an 8-page academic or senior technical report on US Letter with 2cm margins, follow this exact page-by-page budget. It adapts to **any** applied ML domain (tabular, time series, NLP, vision, recommender systems) while preventing trailing 9th-page overflows:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Page 1: Title Header | Executive Summary | Section 1 (Problem Scale)   │
├────────────────────────────────────────────────────────────────────────┤
│ Page 2: Top-of-Page Visual Hook (Fig 1) | Section 2 (Data Description) │
├────────────────────────────────────────────────────────────────────────┤
│ Page 3: Data Cleaning (Tables 2 & 3) | Feature Extraction Intro        │
├────────────────────────────────────────────────────────────────────────┤
│ Page 4: Mathematical Feature Formulations (Equations 1, 2, 3)          │
├────────────────────────────────────────────────────────────────────────┤
│ Page 5: Section 3 (Methodology, Convergence Proofs, Runtime SLA)       │
├────────────────────────────────────────────────────────────────────────┤
│ Page 6: Section 4 (Results Intro, Model Specs Table 4, Diagnostics)    │
├────────────────────────────────────────────────────────────────────────┤
│ Page 7: Section 4.3 (Operational Lift Table 6) | Section 5 Conclusions │
├────────────────────────────────────────────────────────────────────────┤
│ Page 8: Section 6 (Complete Academic References [1] to [9])            │
└────────────────────────────────────────────────────────────────────────┘
```

### Detailed Page Allocation:
* **Page 1:** Minimalist Title Header (Title, `António Barroso`, `Month, Year`), 7–8 line standalone **Executive Summary** (`\small\parbox`), Section 1 (1.1 Problem Scale & Decision Stakes, 1.2 Telemetry Harvesting, 1.3 Heterogeneity in Sub-Domains). Concludes with a forward reference to Figure 1. *Layout Anchor:* `\enlargethispage{3\baselineskip}`, ends with `\clearpage`.
* **Page 2:** **Figure 1** at the top (tri-panel profile of interaction patterns across domain roles/classes). Section 2.1 Data Description, **Table 1: Feature Description Dictionary**, Section 2.2 Data Segregation (entity-level partitioning to prevent data leakage). Ends with `\clearpage`.
* **Page 3:** **Table 2: Data Overview Before Cleaning**, Section 2.3 Data Cleaning (2.3.1 Domain Anomaly 1, 2.3.2 Domain Anomaly 2), **Table 3: Data Overview After Cleaning**, Section 2.4 Extra Feature Extraction Intro. Ends with `\clearpage`.
* **Page 4:** Detailed feature engineering with formal equations:
  - Eq. 1: Residualization / Baseline Normalization / Expectation marginalization.
  - Eq. 2: Statistical Regularization / Empirical Bayes Shrinkage.
  - Eq. 3: Domain Constraint / Context Penalty / Guardrail Index.
  Ends with `\clearpage`.
* **Page 5:** **Section 3: Methodology & System Architecture** (NEVER in Results):
  - 3.1 Mathematical Loss Formulation (Eq. 4).
  - 3.2 Engineering Tradeoffs & Convergence Proofs (Eq. 5): Sample size asymptotic standard error derivation ($1/\sqrt{N}$ law), variance shrinkage tuning, deadband storage pruning, and execution latency.
  - 3.3 Unified Runtime Decision Evaluator (Eq. 6).
  Ends with `\clearpage`.
* **Page 6:** **Section 4: Empirical Results & Model Performance** (Strictly Empirical):
  - Section 4 Intro & **Table 4: Model Specifications Overview**.
  - 4.1 Discussion of Model Performance & **Table 5: Diagnostics Across Cohorts** (Accuracy, ROC-AUC, Log-Loss, Brier Score).
  - 4.2 Statistical Inference & Tactical Role Drivers (Parameter coefficients $\beta$, $z$-scores, $p$-values, covariance analysis).
  *Layout Anchor:* All role drivers fit completely on Page 6. Ends with `\clearpage`.
* **Page 7:** **Section 4.3 Operational Lift & Section 5 Conclusions:**
  - 4.3 Outcome Lift: Model-Assisted vs. Unassisted Baseline.
  - **Table 6: Empirical Outcomes by Advantage Tier / Quintile** (quantifying real-world win rate / conversion lift).
  - Sub-cohort impact analysis.
  - **Section 5: Conclusions and Future Work** (empirical summary, quantified lift, realistic research avenues).
  Ends with `\clearpage`.
* **Page 8:** **Section 6: References** ([1] through [9]), cleanly centered with balanced vertical spacing (`\itemsep{1.2em}`).

---

## 4. Taxonomy of Sections: What Belongs Where

### Front Matter: Executive Summary Box
* **Location:** Top of Page 1, immediately below the author/date line and preceding Section 1.
* **Format:** Formatted using `\noindent\parbox{\textwidth}{\small\textbf{Executive Summary} --- ...}`.
* **Mandatory Ingredients:**
  1. *Problem Scale & Decision Stakes:* Astronomical state space or operational volume under tight real-time constraints.
  2. *Data Scope:* Total verified observation volume, target population tier, and regional/cohort sources.
  3. *Core ML Formulation:* Feature transformations, regularization method, and asymptotic convergence sweet spot.
  4. *Quantified Operational Lift:* Empirical outcome swing between unassisted baseline and model-optimized cases (in percentage points).
  5. *Production SLA:* Sub-millisecond scoring latency (<0.05ms) or throughput benchmark.

### Section 1: Problem Scale & Operational Motivation
* Lead with the **exact scale of the problem** (combinatoric states, data volume, transaction velocity) and why human intuition fails under time constraints (e.g., 30s timers, live streaming updates).
* Justify the data sampling tier: why filtering out low-tier noise preserves strategic fidelity while maintaining sufficient volume across niche interactions.
* Call out Figure 1 at the end of Section 1 to bridge visually to Page 2.

### Section 2: Data Hygiene, Leakage Prevention & Feature Engineering
* **Entity-Level Partitioning:** Always explain why random train/test splits cause data leakage when observations are coupled (by match, user, vessel, or session), and document the entity-level split.
* **Cleaning Auditing:** Tables 2 and 3 must account for every pruned observation with concrete domain justifications (remakes, disconnects, telemetry dropouts).
* **Residual Transformations:** Never feed raw collinear features into linear models; transform features into zero-centered residual deltas relative to baselines.
* **Shrinkage Regularization:** Address rare interactions with Empirical Bayes shrinkage toward prior means to eliminate low-sample noise.

### Section 3: Methodology & Engineering Tradeoffs (Methodology Domain)
* **Mathematical Objective:** Present the loss function (Eq. 4) with explicit justification for why this model family was chosen (e.g., odds-ratio interpretability, convex convergence).
* **Convergence Proofs & Tradeoffs (Section 3.2):**
  - **Asymptotic Square-Root Law:** Derive $\text{SE}(\hat{\beta}) \approx 1/\sqrt{N}$ from M-estimation theory. Show the point estimate delta curve to prove why the selected sample size represents the optimal sweet spot before hitting API quotas or diminishing returns.
  - **Compression & Latency:** Document deadband pruning and memory footprint reductions.
* **Unified Decision Evaluator:** Detail the final composite scoring function (Eq. 6).

### Section 4: Results & Model Performance (Empirical Domain Only)
* **Strict Empirical Boundary:** Never put system architecture, latency proofs, or theoretical derivations in Section 4. Section 4 contains **only test set outputs**.
* **Diagnostics (Table 5):** Report probabilistic metrics (ROC-AUC, Log-Loss, Brier score) alongside accuracy, especially for balanced classes where accuracy alone is uninformative.
* **Statistical Inference (Section 4.2):** Report asymptotic covariance outputs ($\beta$, $z$, $p$-values) and explain the domain forces driving each sub-cohort.
* **Operational Lift (Table 6):** Partition outcomes into quintiles or advantage tiers to prove that model recommendations translate directly to practical business/game victory.

### Section 5: Conclusions and Future Work
* Summarize the empirical findings, restate the quantified operational lift (+X pp), and propose realistic extensions (e.g., graph neural networks, temporal objective trajectories).

---

## 5. Authentic Phrasing and Syntactic Templates

Use these authentic syntactic constructions when generating technical reports in this style:

### A. Executive Summary
- *"Executive Summary --- In [domain], [task] spans over [N] combinations across [M] [entities], where unassisted deficits swing [outcome] by [X]% to [Y]%. This paper presents an interpretable, calibrated machine learning decision-support system for real-time [task] optimization. Using [N] observations from [data source] across [regions], we engineer [features] and [smoothing method]. We train position-specific regularized models, prove asymptotic sample size convergence ($1/\sqrt{N}$ law), and benchmark a sub-[T]ms decision function. In empirical validation across [cohorts/quintiles], model-optimized [decisions] achieve an observed [Z]% [outcome rate] versus [W]% for unassisted deficit cases (+[D] percentage point lift)."*

### B. Problem Scale & Domain Scope (Section 1)
- *"With over [N] [entities] across [M] positions, there are around [X] viable combinations. Manually evaluating all possible pairings becomes unfeasible under tight [T]-second timers."*
- *"Through machine learning, analysts and coaching staff may have at their disposal a better and more cost-efficient tool to countermeasure this problem."*
- *"One of the great things about [telemetry] is that it is openly accessible and provides real-time information about [entity] behavior."*
- *"Fig. 1 shows how [entity] interaction profiles differ based on [category]. It is important to assess whether our data can capture [category] pattern diverseness."*

### C. Data Cleaning & Feature Transformation (Section 2)
- *"Arranging and cleaning [X] telemetry is not a simple task. Hence, this report provides an extensive outlook on the data preparation part of our model."*
- *"In order to get the most out of [X] telemetry, we need to overcome some of its limitations and perform additional feature extraction. The first part of the section expands on the process of cleaning up the data."*
- *"After both procedures, a total of [N] instances ([M] observations) were removed."*
- *"We suspected that absolute values for [feature] would lead to overfitting. We incur in the risk of training our model to make predictions only for [subset], completely failing to generalize. Thus, this feature needs to be transformed for real-world applicability."*
- *"Our solution was to track [X] by computing zero-centered residual deltas between consecutive interactions... As a result, our model was able to make use of how [X] affects (or does not affect) [target]."*
- *"When observations $N$ are scarce, the estimated advantage shrinks toward zero, effectively eliminating statistical noise."*
- *"Amid of the procedures described above, we ended up adding [N] new features [...] and modifying 1 ([feature])."*

### D. Methodology & Convergence Tradeoffs (Section 3)
- *"In this classification task, we are interested in obtaining the highest accuracy with the least amount of false negatives, while preserving strict parameter interpretability and avoiding overfitting."*
- *"Setting these parameters offers an opportunity to consider the trade-off between model generalizability and parameter shrinkage."*
- *"Crucially, the logistic formulation guarantees that every learned parameter $\beta_j$ directly translates to a multiplicative change in win odds ($e^{\beta_j}$), preserving full tactical interpretability for coaching staff."*
- *"Under asymptotic M-estimation theory, parameter standard errors scale inversely with sample size ($\text{SE}(\hat{\beta}) \approx 1/\sqrt{N}$)... Given that live decision margins average [A] to [B] points, crawling beyond [N] observations produces zero practical ranking flips while consuming over [H] additional hours of [resource]. Hence, [N] observations represents the optimal convergence sweet spot."*

### E. Results & Tactical Inferences (Section 4)
- *"In this section, we ought to evaluate the performance of our models."*
- *"Because [target] classes are balanced 50/50, traditional sensitivity and specificity metrics provide redundant information... For these reasons, we evaluated model performance through Receiver Operating Characteristic Area Under the Curve (ROC-AUC), thresholded accuracy, and binary cross-entropy log-loss."*
- *"Examining the asymptotic covariance matrix reveals the forces governing each role: [Feature] is the single strongest isolated predictor across the map ($z = +6.21, p < 0.0001, \beta = +0.0735$)."*
- *"To measure operational utility in practice, we evaluated match victory rates across quintiles of collective draft advantage (Table 6). In unassisted drafts, players frequently fall into severe traps... In contrast, when compositions align with our model's top recommendations, win rate surges to 58.3%: an empirical swing of 19.6 percentage points purely attributable to draft quality."*
- *"We concluded, based on the posterior results, that using [Model] on the altered features provided us with the best model as measured by different accuracy metrics and training time."*

### F. Conclusions (Section 5)
- *"Machine learning makes it possible to develop effective tools to tackle [domain] issues. This report presented a modest application of machine learning methods to face the challenges of [problem]."*
- *"Based on our findings, we concluded that initial [X] features can be modified to create new ones and increase model performance."*
- *"For future work we may consider expanding the effects of [temporal / architectural element] or using neural networks and graph convolutional models to analyze [complex representation] for predictions."*

---

## 6. Technical LaTeX Formatting & Compilation Guidelines

* **Compiler:** Use `tectonic <file>.tex -o <dir>` or `xelatex`.
* **Geometry:** `\usepackage[margin=2cm]{geometry}` on `11pt, letterpaper`.
* **Table Design Discipline:**
  - Never allow wide feature strings to wrap redundantly across 5+ individual rows. Group identical cohorts on unified rows to preserve vertical space.
  - Set `\setlength{\tabcolsep}{5pt}` or `6pt` and use compact column headers (`Obs. Win Rate` instead of `Observed Win Rate`) to eliminate overfull `\hbox` warnings.
* **Vertical Page Control:**
  - Use `\enlargethispage{2\baselineskip}` or `\enlargethispage{3\baselineskip}` deliberately on dense pages (Pages 1, 2, 5, 6, 7).
  - Use `\clearpage` at the exact boundary of each page budget to enforce deterministic 8-page rendering.
* **Verification Protocol:** Always inspect page count programmatically via PDF object stream decompression (`pypdf` or zlib stream parsing) to verify **strictly 8 pages**. Never trust command return codes alone.

import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

df = pd.read_csv("data/training_dataset.csv")

# side: 0 for blue, 1 for red
df["side"] = df["side"].map({"blue": 0, "red": 1})

# Make sure a single match is not split across train and test
unique_matches = df["match_id"].drop_duplicates()
train_matches, test_matches = train_test_split(unique_matches, test_size=0.2, random_state=67)

train_df = df[df["match_id"].isin(train_matches)]
test_df = df[df["match_id"].isin(test_matches)]

# We need a model for each role
roles = ['top', 'jungle', 'middle', 'bottom', 'support']
models = {}
summary_rows = []
inference_rows = []

for role in roles:
    train_role_df = train_df[train_df["role"] == role]
    test_role_df = test_df[test_df["role"] == role]

    # Common features to all lanes
    role_features = ["side", "baseline_wr", "lane_delta", "team_synergy", "threat_delta"]

    # Bot and support have duo partner synergy
    if role in ["bottom", "support"]:
        role_features.append("duo_synergy")

    x_train = train_role_df[role_features]
    y_train = train_role_df["win"]
    x_test = test_role_df[role_features]
    y_test = test_role_df["win"]

    # Fit L2 regularized logistic model
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    model.fit(x_train, y_train)
    models[role] = model

    # Evaluate on test set
    test_preds = model.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, (test_preds >= 0.5).astype(int))
    auc = roc_auc_score(y_test, test_preds)
    loss = log_loss(y_test, test_preds)
    brier = brier_score_loss(y_test, test_preds)

    # Compute asymptotic covariance matrix via Fisher Information for p-values & SEs
    X_design = np.column_stack([np.ones(len(x_train)), x_train.values])
    train_preds = model.predict_proba(x_train)[:, 1]
    W = train_preds * (1 - train_preds)
    cov_matrix = np.linalg.inv(np.dot(X_design.T * W, X_design))
    standard_errors = np.sqrt(np.diagonal(cov_matrix))[1:]  # Skip intercept
    z_scores = model.coef_[0] / standard_errors
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    ci_lower = model.coef_[0] - 1.96 * standard_errors
    ci_upper = model.coef_[0] + 1.96 * standard_errors

    print(f"\n=== {role.upper()} ===")
    print(f"Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f} | Log-Loss: {loss:.4f} | Brier: {brier:.4f}")
    weights = dict(zip(role_features, model.coef_[0].round(4)))
    print("Calibrated Weights:", weights)

    summary_rows.append({
        "Role": role.upper(),
        "ROC-AUC": round(auc, 3),
        "Accuracy": f"{acc * 100:.1f}%",
        "Log-Loss": round(loss, 4),
        "Brier": round(brier, 4),
        "side": weights.get("side", 0.0),
        "baseline_wr": weights.get("baseline_wr", 0.0),
        "lane_delta": weights.get("lane_delta", 0.0),
        "team_synergy": weights.get("team_synergy", 0.0),
        "threat_delta": weights.get("threat_delta", 0.0),
        "duo_synergy": weights.get("duo_synergy", 0.0),
    })

    for f_name, c_val, s_val, z_val, p_val, l_val, u_val in zip(
        role_features, model.coef_[0], standard_errors, z_scores, p_values, ci_lower, ci_upper
    ):
        sig_star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
        inference_rows.append({
            "Role": role.upper(),
            "Feature": f_name,
            "Coef (Beta)": f"{c_val:+.4f}",
            "Std.Err": f"{s_val:.4f}",
            "z-stat": f"{z_val:+.2f}",
            "p-value": p_str,
            "Sig": sig_star,
            "95% CI": f"[{l_val:+.3f}, {u_val:+.3f}]"
        })

# 1. Performance Diagnostics Table
perf_df = pd.DataFrame(summary_rows)[["Role", "ROC-AUC", "Accuracy", "Log-Loss", "Brier"]]
print("\n" + "=" * 85)
print("            MODEL DIAGNOSTICS & PREDICTIVE ACCURACY")
print("=" * 85)
print(perf_df.to_string(index=False))

# 2. Comprehensive Statistical Inference Table (SE, z, p-values, 95% CI)
inf_df = pd.DataFrame(inference_rows)
print("\n" + "=" * 85)
print("         STATISTICAL INFERENCE TABLE (STANDARD ERRORS, Z-STATS, P-VALUES)")
print("         Significance codes: '***' p<0.001 | '**' p<0.01 | '*' p<0.05 | 'ns' p>=0.05")
print("=" * 85)
print(inf_df.to_string(index=False))

# 3. Normalized Scoring Multipliers for DraftScorer (Base = 1.00)
norm_rows = []
for r in summary_rows:
    base = float(r["baseline_wr"]) if float(r["baseline_wr"]) > 0 else 0.015
    norm_rows.append({
        "Role": r["Role"],
        "base": 1.00,
        "lane": round(float(r["lane_delta"]) / base, 2),
        "synergy": round(float(r["team_synergy"]) / base, 2),
        "threat": round(float(r["threat_delta"]) / base, 2),
        "duo_synergy": round(float(r["duo_synergy"]) / base, 2),
    })

norm_df = pd.DataFrame(norm_rows)
print("\n" + "=" * 85)
print("       NORMALIZED SCORING MULTIPLIERS (FOR DRAFTSCORER, BASE = 1.00)")
print("=" * 85)
print(norm_df.to_string(index=False))
print("=" * 85 + "\n")


"""
===============================================================================
FINAL CALIBRATED MODEL RESULTS (4,145 MATCHES / 41,450 SAMPLES)
Balanced across KR (1,631), EUW (1,308), NA (1,200)
===============================================================================

1. TOP (Accuracy: 51.1% | ROC-AUC: 0.515)
   - Primary Drivers: lane_delta (+0.0735, p < 0.0001 ***), team_synergy (+0.0443, p < 0.0001 ***)
   - Takeaway: Lane counter advantage is the single strongest isolated 1v1 predictor (z = 6.21).
   - Normalized Scoring Weights: base: 1.00, lane: 1.58, synergy: 0.95, threat: 0.55

2. JUNGLE (Accuracy: 51.3% | ROC-AUC: 0.511)
   - Primary Drivers: threat_delta (+0.0485, p < 0.0001 ***), lane_delta (+0.0561, p < 0.0001 ***), team_synergy (+0.0372, p = 0.0003 ***)
   - Takeaway: Threat management and cross-lane matchup coverage dominate jungle impact.
   - Normalized Scoring Weights: base: 1.00, lane: 2.18, synergy: 1.45, threat: 1.89

3. MIDDLE (Accuracy: 51.1% | ROC-AUC: 0.512)
   - Primary Drivers: threat_delta (+0.0505, p < 0.0001 ***), lane_delta (+0.0460, p < 0.0001 ***), team_synergy (+0.0402, p < 0.0001 ***)
   - Takeaway: All 3 compositional dimensions are highly statistically significant (p < 0.0001).
   - Normalized Scoring Weights: base: 1.00, lane: 1.80, synergy: 1.58, threat: 1.98

4. BOTTOM / ADC (Accuracy: 50.7% | ROC-AUC: 0.523)
   - Primary Drivers: baseline_wr (+0.0450, p = 0.0042 **), lane_delta (+0.0438, p = 0.0004 ***)
   - Takeaway: ADC outcomes depend on raw champion baseline strength and 2v2 lane survival; team synergy is insignificant (p = 0.11).
   - Normalized Scoring Weights: base: 1.00, lane: 0.97, synergy: 0.42, threat: 0.50, duo_synergy: 0.63

5. SUPPORT (Accuracy: 50.6% | ROC-AUC: 0.516)
   - Primary Drivers: threat_delta (+0.0413, p < 0.0001 ***), duo_synergy (+0.0324, p = 0.0221 *), lane_delta (+0.0328, p = 0.0091 **)
   - Takeaway: Roaming threat mitigation and anchoring bot duo partner are primary drivers.
   - Normalized Scoring Weights: base: 1.00, lane: 0.70, synergy: 0.51, threat: 0.88, duo_synergy: 0.69

RED SIDE LEVERAGE:
   - Consistent +0.15 to +0.20 log-odds across all roles, reflecting R5 counter-pick privilege in Emerald+ draft.
===============================================================================
"""



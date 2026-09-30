#!/usr/bin/env python3
"""
LolDraft Role Inference & Draft Scoring Engine
Implements Bayesian marginal role inference over locked enemy picks using exact permutation likelihoods,
computes expected-value matchup scoring during live Champion Select, and enforces compositional damage guardrails.
"""

from itertools import permutations
from typing import Dict, List, Any, Optional

ALL_ROLES = ["top", "jungle", "middle", "bottom", "support"]

class RoleInferenceEngine:
    """
    Computes exact marginal role probabilities P(enemy_i = role_r) across locked enemy champions.
    """

    EPSILON = 0.005  # Base floor probability for off-meta flexibility

    def infer_roles(
        self,
        locked_enemy_cids: List[str],
        matrix: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates exact marginal probability distribution for each enemy champion.
        Returns mapping: { cid: { "top": prob, "jungle": prob, ... } }
        """
        if not locked_enemy_cids:
            return {}

        champions_data = matrix.get("champions", {})
        k = len(locked_enemy_cids)

        # Collect priors for each enemy champion
        enemy_priors = {}
        for cid in locked_enemy_cids:
            champ = champions_data.get(str(cid), {})
            priors = champ.get("role_priors", {})
            dist = {}
            for r in ALL_ROLES:
                raw_p = priors.get(r, 0.0)
                dist[r] = max(raw_p, self.EPSILON)
            total = sum(dist.values())
            enemy_priors[cid] = {r: dist[r] / total for r in ALL_ROLES}

        if k == 1:
            return enemy_priors

        # Sum likelihood over all permutations of roles for k champions
        total_likelihood = 0.0
        role_joint_sums = {cid: {r: 0.0 for r in ALL_ROLES} for cid in locked_enemy_cids}

        for role_assignment in permutations(ALL_ROLES, k):
            assignment_prob = 1.0
            for i, cid in enumerate(locked_enemy_cids):
                assigned_role = role_assignment[i]
                assignment_prob *= enemy_priors[cid][assigned_role]

            total_likelihood += assignment_prob
            for i, cid in enumerate(locked_enemy_cids):
                assigned_role = role_assignment[i]
                role_joint_sums[cid][assigned_role] += assignment_prob

        # Normalize marginals
        results = {}
        for cid in locked_enemy_cids:
            results[cid] = {}
            for r in ALL_ROLES:
                prob = role_joint_sums[cid][r] / total_likelihood if total_likelihood > 0 else 0.2
                results[cid][r] = round(prob, 4)

        return results


class DraftScorer:
    """
    Scores champion candidates for a specific player role considering turn context,
    Bayesian role expectations, ally synergies, blind vulnerability, and compositional guardrails.
    """

    def __init__(self, matrix: Dict[str, Any]):
        self.matrix = matrix
        self.inference_engine = RoleInferenceEngine()
        self.weights = matrix.get("weights", {
            "base": 1.0,
            "lane": 1.35,
            "synergy": 0.85,
            "threat": 0.60,
            "blind": 0.90
        })

    def get_champion_damage_profile(self, cid: str, preferred_role: Optional[str] = None) -> Dict[str, float]:
        """
        Retrieves the damage profile percentages for a champion, defaulting to their primary role.
        """
        champ = self.matrix.get("champions", {}).get(str(cid), {})
        roles = champ.get("roles", {})
        if preferred_role and preferred_role in roles:
            return roles[preferred_role].get("damage_profile", {"pct_physical": 50.0, "pct_magic": 50.0, "pct_true": 0.0})

        priors = champ.get("role_priors", {})
        top_role = max(priors.items(), key=lambda x: x[1])[0] if priors else (list(roles.keys())[0] if roles else None)
        if top_role and top_role in roles:
            return roles[top_role].get("damage_profile", {"pct_physical": 50.0, "pct_magic": 50.0, "pct_true": 0.0})

        return {"pct_physical": 50.0, "pct_magic": 50.0, "pct_true": 0.0}

    def score_candidate(
        self,
        candidate_cid: str,
        assigned_role: str,
        locked_allies: List[str],
        locked_enemies: List[str]
    ) -> Dict[str, Any]:
        """
        Computes composite draft score, explainable rationale, and damage balance for a candidate pick.
        """
        champions = self.matrix.get("champions", {})
        cand = champions.get(str(candidate_cid))
        if not cand:
            return {"error": f"Unknown champion ID {candidate_cid}"}

        cand_name = cand.get("name", "Unknown")
        roles = cand.get("roles", {})

        if assigned_role not in roles:
            return {
                "cid": candidate_cid,
                "name": cand_name,
                "role": assigned_role,
                "viable": False,
                "score": -999.0,
                "rationale": [f"Off-meta: No verified high-ELO data for {cand_name} in {assigned_role.upper()}."]
            }

        role_data = roles[assigned_role]
        baseline_wr = role_data.get("baseline_wr", 50.0)
        blind_vuln = role_data.get("blind_vulnerability", 1.5)
        cand_dmg = role_data.get("damage_profile", {"pct_physical": 50.0, "pct_magic": 50.0, "pct_true": 0.0})
        counters = role_data.get("counters", {})
        threats = role_data.get("threats", {})
        synergies = role_data.get("synergies", {})

        # Step 1: Infer enemy role probabilities
        enemy_roles = self.inference_engine.infer_roles(locked_enemies, self.matrix)

        # Step 2: Compute expected enemy matchup deltas
        expected_lane_delta = 0.0
        expected_threat_delta = 0.0
        matchup_breakdown = []
        total_lane_prob = 0.0

        for e_cid in locked_enemies:
            e_name = champions.get(str(e_cid), {}).get("name", f"Enemy #{e_cid}")
            p_lane = enemy_roles.get(str(e_cid), {}).get(assigned_role, 0.0)
            total_lane_prob += p_lane

            lane_delta = counters.get(str(e_cid), 0.0)
            threat_delta = threats.get(str(e_cid), 0.0)

            expected_lane_delta += p_lane * lane_delta
            expected_threat_delta += (1.0 - p_lane) * threat_delta

            matchup_breakdown.append({
                "enemy_cid": e_cid,
                "enemy_name": e_name,
                "p_lane": round(p_lane, 2),
                "lane_delta": lane_delta,
                "threat_delta": threat_delta
            })

        total_lane_prob = min(total_lane_prob, 1.0)

        # Step 3: Compute ally synergies
        total_synergy_delta = 0.0
        synergy_breakdown = []
        for a_cid in locked_allies:
            a_name = champions.get(str(a_cid), {}).get("name", f"Ally #{a_cid}")
            syn_delta = synergies.get(str(a_cid), 0.0)
            total_synergy_delta += syn_delta
            synergy_breakdown.append({
                "ally_cid": a_cid,
                "ally_name": a_name,
                "delta": syn_delta
            })

        # Step 4: Compositional Guardrails (Damage profile balance)
        comp_adjustment = 0.0
        rationale = []
        projected_phys = cand_dmg.get("pct_physical", 50.0)
        projected_magic = cand_dmg.get("pct_magic", 50.0)

        if locked_allies:
            allies_dmg = [self.get_champion_damage_profile(a_cid) for a_cid in locked_allies]
            all_dmg_list = allies_dmg + [cand_dmg]
            projected_phys = sum(d["pct_physical"] for d in all_dmg_list) / len(all_dmg_list)
            projected_magic = sum(d["pct_magic"] for d in all_dmg_list) / len(all_dmg_list)

            # Check if existing allies already heavily skew towards one damage type
            existing_phys = sum(d["pct_physical"] for d in allies_dmg) / len(allies_dmg)
            existing_magic = sum(d["pct_magic"] for d in allies_dmg) / len(allies_dmg)

            if len(locked_allies) >= 2:
                # Heavy Physical Damage Warning (>= 75% physical among locked allies)
                if existing_phys >= 75.0:
                    if cand_dmg.get("pct_physical", 0) >= 65.0:
                        comp_adjustment -= 3.0
                        rationale.append("Composition Penalty: Heavy Physical redundancy (-3.00%). Enemy team can easily stack Armor.")
                    elif cand_dmg.get("pct_magic", 0) >= 60.0:
                        comp_adjustment += 2.5
                        rationale.append("Composition Bonus: Critical AP diversification (+2.50%). Prevents enemy Armor stacking.")

                # Heavy Magic Damage Warning (>= 75% magic among locked allies)
                elif existing_magic >= 75.0:
                    if cand_dmg.get("pct_magic", 0) >= 65.0:
                        comp_adjustment -= 3.0
                        rationale.append("Composition Penalty: Heavy Magic redundancy (-3.00%). Enemy team can easily stack Magic Resist.")
                    elif cand_dmg.get("pct_physical", 0) >= 60.0:
                        comp_adjustment += 2.5
                        rationale.append("Composition Bonus: Critical AD diversification (+2.50%). Prevents enemy Magic Resist stacking.")

        # Step 5: Turn Context Scoring (Blind pick vs Revealed counter)
        is_blind = total_lane_prob < 0.25
        blind_penalty = 0.0
        if is_blind:
            unrevealed_factor = 1.0 - total_lane_prob
            blind_penalty = self.weights["blind"] * blind_vuln * unrevealed_factor
            if blind_vuln < 2.0:
                rationale.append(f"Safe Blind: Low vulnerability rating ({blind_vuln:.1f}% avg counter severity).")
            else:
                rationale.append(f"Risky Blind: Punished hard by counters ({blind_vuln:.1f}% avg counter severity).")
        else:
            if expected_lane_delta > 1.0:
                rationale.append(f"Favorable Lane Matchup (+{expected_lane_delta:.2f}% expected delta).")
            elif expected_lane_delta < -1.0:
                rationale.append(f"Unfavorable Lane Matchup ({expected_lane_delta:.2f}% expected delta).")

        if total_synergy_delta > 0.8:
            rationale.append(f"Strong Team Synergy (+{total_synergy_delta:.2f}%).")
        elif total_synergy_delta < -0.8:
            rationale.append(f"Negative Team Synergy ({total_synergy_delta:.2f}%).")

        composite_score = (
            self.weights["base"] * baseline_wr
            + self.weights["lane"] * expected_lane_delta
            + self.weights["threat"] * expected_threat_delta
            + self.weights["synergy"] * total_synergy_delta
            + comp_adjustment
            - blind_penalty
        )

        return {
            "cid": candidate_cid,
            "name": cand_name,
            "role": assigned_role,
            "viable": True,
            "composite_score": round(composite_score, 2),
            "baseline_wr": baseline_wr,
            "expected_lane_delta": round(expected_lane_delta, 2),
            "expected_threat_delta": round(expected_threat_delta, 2),
            "synergy_delta": round(total_synergy_delta, 2),
            "composition_adjustment": round(comp_adjustment, 2),
            "blind_penalty": round(blind_penalty, 2),
            "lane_opponent_revealed_prob": round(total_lane_prob, 2),
            "damage_profile": cand_dmg,
            "projected_team_damage": {
                "pct_physical": round(projected_phys, 1),
                "pct_magic": round(projected_magic, 1)
            },
            "rationale": rationale,
            "matchups": matchup_breakdown,
            "synergies": synergy_breakdown
        }

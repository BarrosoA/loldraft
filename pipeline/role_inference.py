#!/usr/bin/env python3
"""
LolDraft Role Inference & Draft Scoring Engine
Implements Bayesian marginal role inference over locked enemy picks using exact permutation likelihoods,
computes expected-value matchup scoring during live Champion Select, and enforces compositional damage guardrails.
"""

from itertools import permutations
from typing import Dict, List, Any, Optional

ALL_ROLES = ["top", "jungle", "middle", "bottom", "support"]

# Recognized Armor-scaling / Heavy Vanguard Tank Champions
ARMOR_STACKING_CHAMPIONS = {
    '54', '33', '897', '78', '44', '516', '14', '89', '111', '98', '31', '154', '113', '57', '223', '201', '12'
}

MR_STACKING_CHAMPIONS = {
    '38', '3', '27', '516', '57', '32', '86'
}

ROLE_AP_PRIORS = {
    'middle': 0.68,
    'jungle': 0.28,
    'top': 0.24,
    'bottom': 0.05
}

ROLE_AD_PRIORS = {
    'middle': 0.32,
    'jungle': 0.72,
    'top': 0.76,
    'bottom': 0.95
}

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
            "duo_synergy": 1.35,
            "threat": 0.60,
            "blind": 0.90
        })
        if "duo_synergy" not in self.weights:
            self.weights["duo_synergy"] = 1.35

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
        locked_enemies: List[str],
        ally_roles: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Computes composite draft score, explainable rationale, and damage balance for a candidate pick.
        """
        champions = self.matrix.get("champions", {})
        cand = champions.get(str(candidate_cid))
        if not cand:
            return {"error": f"Unknown champion ID {candidate_cid}"}

        if ally_roles is None:
            ally_roles = {}
            for a_cid in locked_allies:
                priors = champions.get(str(a_cid), {}).get("role_priors", {})
                if priors:
                    ally_roles[str(a_cid)] = max(priors.items(), key=lambda x: x[1])[0]

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

        # Step 3: Compute ally synergies (distinguishing 2v2 duo lane partner from off-lane team allies)
        total_team_synergy_delta = 0.0
        duo_synergy_delta = 0.0
        synergy_breakdown = []
        duo_partner_info = None

        target_duo_role = None
        if assigned_role == "bottom":
            target_duo_role = "support"
        elif assigned_role in ("support", "utility"):
            target_duo_role = "bottom"

        for a_cid in locked_allies:
            a_name = champions.get(str(a_cid), {}).get("name", f"Ally #{a_cid}")
            syn_delta = synergies.get(str(a_cid), 0.0)
            a_role = ally_roles.get(str(a_cid)) if ally_roles else None

            is_duo = False
            if target_duo_role is not None and duo_partner_info is None:
                if target_duo_role == "support" and a_role in ("support", "utility"):
                    is_duo = True
                elif target_duo_role == "bottom" and a_role == "bottom":
                    is_duo = True

            if is_duo:
                duo_synergy_delta += syn_delta
                duo_partner_info = {"name": a_name, "delta": syn_delta}
                synergy_breakdown.append({
                    "ally_cid": a_cid,
                    "ally_name": a_name,
                    "delta": syn_delta,
                    "is_duo": True
                })
            else:
                total_team_synergy_delta += syn_delta
                synergy_breakdown.append({
                    "ally_cid": a_cid,
                    "ally_name": a_name,
                    "delta": syn_delta,
                    "is_duo": False
                })

        total_synergy_delta = duo_synergy_delta + total_team_synergy_delta

        # Step 4: Compositional Guardrails (Enemy-Aware Damage Vulnerability)
        comp_adjustment = 0.0
        rationale = []
        projected_phys = cand_dmg.get("pct_physical", 50.0)
        projected_magic = cand_dmg.get("pct_magic", 50.0)

        if locked_allies:
            allies_dmg = [self.get_champion_damage_profile(a_cid) for a_cid in locked_allies]
            all_dmg_list = allies_dmg + [cand_dmg]
            projected_phys = sum(d["pct_physical"] for d in all_dmg_list) / len(all_dmg_list)
            projected_magic = sum(d["pct_magic"] for d in all_dmg_list) / len(all_dmg_list)

            # Determine core carry roles locked vs open
            ally_roles_map = ally_roles or {}
            locked_core_roles = [ally_roles_map.get(str(a_cid)) for a_cid in locked_allies if ally_roles_map.get(str(a_cid)) in ROLE_AP_PRIORS]
            open_core_roles = [r for r in ROLE_AP_PRIORS.keys() if r not in locked_core_roles and r != assigned_role]

            has_locked_ap_carry = any((d.get("pct_magic", 0) >= 50.0) for d in allies_dmg)
            has_locked_ad_carry = any((d.get("pct_physical", 0) >= 50.0) for d in allies_dmg)

            enemy_armor_factor = 0.6
            armor_stacker_names = []
            for e_cid in locked_enemies:
                if str(e_cid) in ARMOR_STACKING_CHAMPIONS:
                    enemy_armor_factor += 0.45
                    e_name = champions.get(str(e_cid), {}).get("name")
                    if e_name:
                        armor_stacker_names.append(e_name)
            enemy_armor_factor = min(enemy_armor_factor, 1.8)

            enemy_mr_factor = 0.6
            mr_stacker_names = []
            for e_cid in locked_enemies:
                if str(e_cid) in MR_STACKING_CHAMPIONS:
                    enemy_mr_factor += 0.45
                    e_name = champions.get(str(e_cid), {}).get("name")
                    if e_name:
                        mr_stacker_names.append(e_name)
            enemy_mr_factor = min(enemy_mr_factor, 1.8)

            cand_is_ad = cand_dmg.get("pct_physical", 0) >= 65.0
            cand_is_ap = cand_dmg.get("pct_magic", 0) >= 55.0

            # Physical Skew Evaluation
            if not has_locked_ap_carry and not cand_is_ap:
                p_zero_ap_risk = 1.0
                if open_core_roles:
                    for r in open_core_roles:
                        p_zero_ap_risk *= (1.0 - ROLE_AP_PRIORS.get(r, 0.25))
                else:
                    p_zero_ap_risk = 1.0

                if p_zero_ap_risk >= 0.20 and len(locked_allies) >= 1:
                    penalty = min(4.5, p_zero_ap_risk * enemy_armor_factor * 3.6)
                    comp_adjustment -= penalty
                    if p_zero_ap_risk >= 0.85:
                        stacker_info = f" into {', '.join(armor_stacker_names)}" if armor_stacker_names else ""
                        rationale.append(f"Draft Trap: Seals Full AD (-{penalty:.2f}%){stacker_info}. Enemy can build pure Armor")
                    else:
                        rationale.append(f"Damage Warning: Heavy AD compounding (-{penalty:.2f}%). Missing primary AP anchor")
            elif not has_locked_ap_carry and cand_is_ap and len(locked_allies) >= 2:
                bonus = min(3.5, enemy_armor_factor * 2.5)
                comp_adjustment += bonus
                rationale.append(f"Composition Anchor: Crucial AP carry (+{bonus:.2f}%). Prevents enemy Armor stacking")

            # Magic Skew Evaluation
            if not has_locked_ad_carry and not cand_is_ad:
                p_zero_ad_risk = 1.0
                if open_core_roles:
                    for r in open_core_roles:
                        p_zero_ad_risk *= (1.0 - ROLE_AD_PRIORS.get(r, 0.70))
                else:
                    p_zero_ad_risk = 1.0

                if p_zero_ad_risk >= 0.20 and len(locked_allies) >= 1:
                    penalty = min(4.5, p_zero_ad_risk * enemy_mr_factor * 3.6)
                    comp_adjustment -= penalty
                    if p_zero_ad_risk >= 0.85:
                        stacker_info = f" into {', '.join(mr_stacker_names)}" if mr_stacker_names else ""
                        rationale.append(f"Draft Trap: Seals Full AP (-{penalty:.2f}%){stacker_info}. Enemy can build pure MR")
                    else:
                        rationale.append(f"Damage Warning: Heavy AP compounding (-{penalty:.2f}%)")
            elif not has_locked_ad_carry and cand_is_ad and len(locked_allies) >= 2:
                bonus = min(3.5, enemy_mr_factor * 2.5)
                comp_adjustment += bonus
                rationale.append(f"Composition Anchor: Crucial AD carry (+{bonus:.2f}%). Prevents enemy Magic Resist stacking")

        # Step 5: Turn Context Scoring (Blind pick vs Revealed counter)
        is_blind = total_lane_prob < 0.25
        blind_penalty = 0.0
        if is_blind:
            unrevealed_factor = 1.0 - total_lane_prob
            blind_penalty = self.weights["blind"] * blind_vuln * unrevealed_factor
            if blind_vuln < 2.0:
                rationale.append(f"Safe Blind: Low vulnerability rating ({blind_vuln:.1f}% avg counter severity)")
            else:
                rationale.append(f"Risky Blind: Punished hard by counters (-{blind_vuln:.1f}% avg)")
        else:
            if expected_lane_delta > 1.0:
                rationale.append(f"Favorable Lane Matchup (+{expected_lane_delta:.2f}% expected delta)")
            elif expected_lane_delta < -1.0:
                rationale.append(f"Unfavorable Lane Matchup ({expected_lane_delta:.2f}% expected delta)")

        if duo_partner_info and abs(duo_partner_info["delta"]) >= 0.5:
            d_val = duo_partner_info["delta"]
            sign = "+" if d_val > 0 else ""
            if d_val > 0:
                rationale.append(f"Bot Duo Synergy ({sign}{d_val:.2f}% with {duo_partner_info['name']})")
            else:
                rationale.append(f"Bot Duo Friction ({sign}{d_val:.2f}% with {duo_partner_info['name']})")

        if total_team_synergy_delta > 0.8:
            rationale.append(f"Strong Team Synergy (+{total_team_synergy_delta:.2f}%)")
        elif total_team_synergy_delta < -0.8:
            rationale.append(f"Negative Team Synergy ({total_team_synergy_delta:.2f}%)")

        duo_weight = self.weights.get("duo_synergy", 1.35)
        team_syn_weight = self.weights.get("synergy", 0.85)

        composite_score = (
            self.weights["base"] * baseline_wr
            + self.weights["lane"] * expected_lane_delta
            + self.weights["threat"] * expected_threat_delta
            + (duo_weight * duo_synergy_delta)
            + (team_syn_weight * total_team_synergy_delta)
            + comp_adjustment
            - blind_penalty
        )

        role_games = role_data.get("games", 0)
        champ_prior = cand.get("role_priors", {}).get(assigned_role, 0.0)
        is_meta = (role_games >= 4000) or (champ_prior >= 0.25)

        if not is_meta:
            fmt_games = f"{role_games/1000:.1f}k" if role_games >= 1000 else str(role_games)
            rationale.insert(0, f"Off-Meta / Specialist ({fmt_games} games, {champ_prior*100:.1f}% presence)")

        return {
            "cid": candidate_cid,
            "name": cand_name,
            "role": assigned_role,
            "viable": True,
            "is_meta": is_meta,
            "games": role_games,
            "role_prior": champ_prior,
            "composite_score": round(composite_score, 2),
            "baseline_wr": baseline_wr,
            "expected_lane_delta": round(expected_lane_delta, 2),
            "expected_threat_delta": round(expected_threat_delta, 2),
            "synergy_delta": round(total_synergy_delta, 2),
            "duo_synergy_delta": round(duo_synergy_delta, 2),
            "team_synergy_delta": round(total_team_synergy_delta, 2),
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

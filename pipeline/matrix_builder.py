#!/usr/bin/env python3
"""
LolDraft Matrix Builder
Compiles raw scraped champion statistics into the structured, optimized lookup matrix (current_matrix.json).
Builds role-specific matrices for each champion, computes empirical Bayes shrinkage,
prunes near-zero noise deltas (< delta_threshold), and attaches empirical role priors.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LolDraftMatrixBuilder")

class MatrixBuilder:
    """
    Transforms scraped champion payloads into normalized pairwise delta matrices structured by role.
    """

    SMOOTHING_M_LANE = 250.0  # Empirical Bayes shrinkage constant for direct lane counters (games)
    SMOOTHING_M_TEAM = 500.0  # Empirical Bayes shrinkage constant for cross-map threats & ally synergies (games)
    DELTA_THRESHOLD = 0.25    # Deadband threshold: prune deltas with absolute value < 0.25%

    def __init__(
        self,
        output_dir: str = "data",
        smoothing_m_lane: float = 250.0,
        smoothing_m_team: float = 500.0,
        delta_threshold: float = 0.25
    ):
        self.output_dir = output_dir
        self.smoothing_m_lane = smoothing_m_lane
        self.smoothing_m_team = smoothing_m_team
        self.delta_threshold = delta_threshold
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "patches"), exist_ok=True)

    def smooth_delta(self, raw_delta: float, sample_size: int, m: Optional[float] = None) -> float:
        """
        Applies empirical Bayes shrinkage towards zero:
        Smoothed = (Raw_Delta * N) / (N + M)
        """
        if sample_size <= 0:
            return 0.0
        smoothing_m = m if m is not None else self.smoothing_m_lane
        return round((raw_delta * sample_size) / (sample_size + smoothing_m), 2)

    def build_matrix_from_scraped_data(
        self,
        scraped_data: Dict[str, Any],
        output_filename: str = "current_matrix.json"
    ) -> Dict[str, Any]:
        """
        Processes dictionary of scraped champions and compiles role-nested current_matrix.json.
        """
        patch = scraped_data.get("patch", "unknown")
        tier = scraped_data.get("tier", "emerald_plus")
        avg_tier_wr = scraped_data.get("avg_tier_wr", 51.5)
        raw_champs = scraped_data.get("champions", {})

        logger.info(f"Building recommendation matrix for Patch {patch} across {len(raw_champs)} champions (delta threshold: {self.delta_threshold}%)...")

        matrix = {
            "patch": patch,
            "tier": tier,
            "avg_tier_wr": avg_tier_wr,
            "smoothing_m_lane": self.smoothing_m_lane,
            "smoothing_m_team": self.smoothing_m_team,
            "smoothing_m": self.smoothing_m_lane,
            "delta_threshold": self.delta_threshold,
            "weights": {
                "base": 1.0,
                "lane": 1.35,
                "synergy": 0.85,
                "duo_synergy": 1.35,
                "threat": 0.60,
                "blind": 0.90
            },
            "champions": {}
        }

        total_entries_stored = 0
        total_entries_pruned = 0

        for slug, champ_data in raw_champs.items():
            info = champ_data.get("info", {})
            cid = str(info.get("cid", ""))
            if not cid:
                continue

            champ_name = "Vi" if slug.lower() == "vi" or cid == "254" else info.get("name", slug.capitalize())
            role_priors = champ_data.get("role_priors", {})

            # Support both new multi-role schema and legacy flat schema
            roles_dict = champ_data.get("roles")
            if not roles_dict:
                default_lane = champ_data.get("stats", {}).get("default_lane", "middle")
                roles_dict = {default_lane: champ_data}
                if not role_priors:
                    role_priors = {default_lane: 1.0}

            processed_roles = {}

            for role_name, role_content in roles_dict.items():
                stats = role_content.get("stats", {})
                enemy_counters = role_content.get("enemy_counters", {})
                ally_synergies = role_content.get("ally_synergies", {})
                baseline_wr = stats.get("win_rate", 50.0)

                # Compute damage profile percentages
                damage_raw = stats.get("damage", {})
                phys = damage_raw.get("physical", 0.0)
                magic = damage_raw.get("magic", 0.0)
                true_dmg = damage_raw.get("true", 0.0)
                total_dmg = phys + magic + true_dmg or 1.0
                damage_profile = {
                    "pct_physical": round((phys / total_dmg) * 100, 1),
                    "pct_magic": round((magic / total_dmg) * 100, 1),
                    "pct_true": round((true_dmg / total_dmg) * 100, 1)
                }

                lane_counters = {}
                threats = {}
                negative_lane_deltas = []

                for enemy_role, matchups in enemy_counters.items():
                    if not isinstance(matchups, list):
                        continue
                    for row in matchups:
                        if len(row) < 6:
                            continue
                        enemy_cid = str(row[0])
                        matchup_wr = float(row[1])
                        sample_size = int(row[5])

                        raw_delta = matchup_wr - baseline_wr

                        if enemy_role == role_name:
                            smoothed = self.smooth_delta(raw_delta, sample_size, self.smoothing_m_lane)
                            if smoothed < 0:
                                negative_lane_deltas.append(abs(smoothed))
                            if abs(smoothed) >= self.delta_threshold:
                                lane_counters[enemy_cid] = smoothed
                                total_entries_stored += 1
                            else:
                                total_entries_pruned += 1
                        else:
                            smoothed = self.smooth_delta(raw_delta, sample_size, self.smoothing_m_team)
                            if abs(smoothed) >= self.delta_threshold:
                                threats[enemy_cid] = smoothed
                                total_entries_stored += 1
                            else:
                                total_entries_pruned += 1

                # Calculate Blind Vulnerability Index for this specific role
                negative_lane_deltas.sort(reverse=True)
                top_counters = negative_lane_deltas[:5]
                blind_vulnerability = round(sum(top_counters) / len(top_counters), 2) if top_counters else 1.0

                # Process ally synergies for this role
                synergies = {}
                for ally_role, team_pairs in ally_synergies.items():
                    if not isinstance(team_pairs, list):
                        continue
                    for row in team_pairs:
                        if len(row) < 6:
                            continue
                        ally_cid = str(row[0])
                        pair_wr = float(row[1])
                        sample_size = int(row[5])

                        raw_synergy_delta = pair_wr - baseline_wr
                        smoothed_synergy = self.smooth_delta(raw_synergy_delta, sample_size, self.smoothing_m_team)

                        if abs(smoothed_synergy) >= self.delta_threshold:
                            synergies[ally_cid] = smoothed_synergy
                            total_entries_stored += 1
                        else:
                            total_entries_pruned += 1

                processed_roles[role_name] = {
                    "baseline_wr": baseline_wr,
                    "games": stats.get("games", 0),
                    "pick_rate": stats.get("pick_rate", 0.0),
                    "ban_rate": stats.get("ban_rate", 0.0),
                    "blind_vulnerability": blind_vulnerability,
                    "damage_profile": damage_profile,
                    "counters": lane_counters,
                    "threats": threats,
                    "synergies": synergies
                }

            matrix["champions"][cid] = {
                "name": champ_name,
                "slug": slug,
                "role_priors": role_priors,
                "roles": processed_roles
            }

        logger.info(f"Matrix optimization complete: {total_entries_stored} significant deltas retained, {total_entries_pruned} near-zero entries pruned.")

        # Save to current_matrix.json
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2)
        logger.info(f"Successfully compiled role-aware matrix with {len(matrix['champions'])} champions to {output_path}")

        # Also archive to patches/
        patch_path = os.path.join(self.output_dir, "patches", f"{patch}_matrix.json")
        with open(patch_path, "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2)
        logger.info(f"Archived patch matrix to {patch_path}")

        return matrix


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LolDraft Matrix Builder CLI")
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Path to cache directory")
    parser.add_argument("--patch", type=str, default=None, help="Target patch to compile from cache")
    parser.add_argument("--m-lane", type=float, default=250.0, help="Empirical Bayes constant for lane counters (default: 250.0)")
    parser.add_argument("--m-team", type=float, default=500.0, help="Empirical Bayes constant for team threats & synergies (default: 500.0)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Pruning deadband threshold in % (default: 0.25)")
    parser.add_argument("--out", type=str, default="current_matrix.json", help="Output JSON filename")

    args = parser.parse_args()

    builder = MatrixBuilder(
        smoothing_m_lane=args.m_lane,
        smoothing_m_team=args.m_team,
        delta_threshold=args.threshold
    )

    # Discover available cached patch
    patch_dir = os.path.join(args.cache_dir, args.patch) if args.patch else None
    if not patch_dir or not os.path.exists(patch_dir):
        subdirs = [d for d in os.listdir(args.cache_dir) if os.path.isdir(os.path.join(args.cache_dir, d))]
        if not subdirs:
            print("No cached patch data found in data/cache. Run ingestion.py first.")
            return
        subdirs.sort(reverse=True)
        patch_dir = os.path.join(args.cache_dir, subdirs[0])
        args.patch = subdirs[0]

    print(f"Compiling cached champions from {patch_dir} for Patch {args.patch}...")
    cached_champions = {}
    for filename in os.listdir(patch_dir):
        if filename.endswith(".json"):
            slug = filename[:-5]
            with open(os.path.join(patch_dir, filename), "r", encoding="utf-8") as f:
                cached_champions[slug] = json.load(f)

    data_payload = {
        "patch": args.patch,
        "tier": "emerald_plus",
        "avg_tier_wr": 51.5,
        "champions": cached_champions
    }

    builder.build_matrix_from_scraped_data(data_payload, output_filename=args.out)


if __name__ == "__main__":
    main()

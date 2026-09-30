#!/usr/bin/env python3
"""
LolDraft Feature Extractor & Tabular Dataset Builder
Transforms crawled 5v5 matches (matches_16.19.jsonl) and pairwise matrix deltas
(current_matrix.json) into a flat tabular training dataset (training_dataset.csv)
ready for Logistic Regression in model.py.
"""

import os
import sys
import json
import csv
import logging
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DatasetBuilder")

MATCHES_FILE = "data/matches_16.19.jsonl"
MATRIX_FILE = "data/current_matrix.json"
OUTPUT_CSV = "data/training_dataset.csv"

ROLE_ORDER = ["top", "jungle", "middle", "bottom", "support"]


def load_matrix(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Matrix file not found: {path}")
    logger.info(f"Loading matrix from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_csv_dataset(matches_path: str, matrix_path: str, output_csv_path: str):
    matrix = load_matrix(matrix_path)
    champs_data = matrix.get("champions", {})

    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Matches file not found: {matches_path}")

    logger.info(f"Parsing matches from {matches_path}...")

    fieldnames = [
        "match_id",
        "server",
        "side",
        "role",
        "cid",
        "champion_name",
        "win",
        "baseline_wr",
        "lane_delta",
        "duo_synergy",
        "team_synergy",
        "threat_delta"
    ]

    total_matches = 0
    total_samples = 0
    skipped_players = 0

    with open(matches_path, "r", encoding="utf-8") as in_f, \
         open(output_csv_path, "w", newline="", encoding="utf-8") as out_f:

        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for line_num, line in enumerate(in_f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                match = json.loads(line)
            except json.JSONDecodeError:
                continue

            match_id = match.get("match_id", f"match_{line_num}")
            server = match.get("server", "unknown")
            blue_team = match.get("blue_team", [])
            red_team = match.get("red_team", [])

            # Map by role for easy 1v1 and ally lookup
            blue_by_role = {p["role"]: p for p in blue_team}
            red_by_role = {p["role"]: p for p in red_team}

            # Must have 5 standard roles per team
            if len(blue_by_role) != 5 or len(red_by_role) != 5:
                continue

            sides = [
                ("blue", blue_team, blue_by_role, red_by_role),
                ("red", red_team, red_by_role, blue_by_role)
            ]

            for side_name, team_players, ally_map, enemy_map in sides:
                for player in team_players:
                    role = player["role"]
                    cid = str(player["cid"])
                    win = 1 if player["win"] else 0

                    champ_obj = champs_data.get(cid)
                    if not champ_obj:
                        skipped_players += 1
                        continue

                    champ_name = champ_obj.get("name", f"Champ_{cid}")
                    role_data = champ_obj.get("roles", {}).get(role)
                    if not role_data:
                        # Off-meta pick without matrix stats
                        skipped_players += 1
                        continue

                    baseline_wr = role_data.get("baseline_wr", 50.0)
                    counters = role_data.get("counters", {})
                    synergies = role_data.get("synergies", {})
                    threats = role_data.get("threats", {})

                    # 1. Direct lane counter matchup
                    enemy_laner_cid = str(enemy_map[role]["cid"])
                    lane_delta = counters.get(enemy_laner_cid, 0.0)

                    # 2. Duo synergy (bot <-> support) & Team synergy
                    duo_synergy = 0.0
                    team_synergy_sum = 0.0

                    if role == "bottom":
                        supp_cid = str(ally_map["support"]["cid"])
                        duo_synergy = synergies.get(supp_cid, 0.0)
                    elif role in ("support", "utility"):
                        bot_cid = str(ally_map["bottom"]["cid"])
                        duo_synergy = synergies.get(bot_cid, 0.0)

                    # Sum team synergy for other allies
                    for other_role, other_player in ally_map.items():
                        if other_role == role:
                            continue
                        # If bot/support, don't double count the duo partner in team synergy
                        if role == "bottom" and other_role in ("support", "utility"):
                            continue
                        if role in ("support", "utility") and other_role == "bottom":
                            continue

                        ally_cid = str(other_player["cid"])
                        team_synergy_sum += synergies.get(ally_cid, 0.0)

                    # 3. Threat delta: sum across all 4 non-lane enemies
                    threat_sum = 0.0
                    for enemy_role, enemy_player in enemy_map.items():
                        if enemy_role == role:
                            continue  # Direct laner is captured by lane_delta
                        e_cid = str(enemy_player["cid"])
                        threat_sum += threats.get(e_cid, 0.0)

                    row = {
                        "match_id": match_id,
                        "server": server,
                        "side": side_name,
                        "role": role,
                        "cid": cid,
                        "champion_name": champ_name,
                        "win": win,
                        "baseline_wr": round(baseline_wr, 4),
                        "lane_delta": round(lane_delta, 4),
                        "duo_synergy": round(duo_synergy, 4),
                        "team_synergy": round(team_synergy_sum, 4),
                        "threat_delta": round(threat_sum, 4)
                    }

                    writer.writerow(row)
                    total_samples += 1

            total_matches += 1

    logger.info(
        f"Dataset generated successfully!\n"
        f"  Total Matches: {total_matches}\n"
        f"  Total Training Samples: {total_samples}\n"
        f"  Skipped Out-of-Matrix Picks: {skipped_players}\n"
        f"  Saved to: {output_csv_path}"
    )


if __name__ == "__main__":
    matches_file = sys.argv[1] if len(sys.argv) > 1 else MATCHES_FILE
    matrix_file = sys.argv[2] if len(sys.argv) > 2 else MATRIX_FILE
    output_file = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_CSV

    build_csv_dataset(matches_file, matrix_file, output_file)

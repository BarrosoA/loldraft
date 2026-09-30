#!/usr/bin/env python3
"""
LolDraft Local Web Server & Simulator Backend
Serves the interactive HTML5 draft simulator and computes real-time recommendations (< 5ms)
using the offline current_matrix.json.
"""

import os
import sys
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.role_inference import RoleInferenceEngine, DraftScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LolDraftServer")

MATRIX_PATH = "data/current_matrix.json"
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")

# Preload matrix into memory
logger.info("Loading recommendation matrix into memory...")
if not os.path.exists(MATRIX_PATH):
    raise FileNotFoundError(f"Matrix file not found at {MATRIX_PATH}. Run ingestion first.")

with open(MATRIX_PATH, "r", encoding="utf-8") as f:
    MATRIX = json.load(f)

SCORER = DraftScorer(MATRIX)
INFERENCE = RoleInferenceEngine()
logger.info(f"Loaded matrix for Patch {MATRIX.get('patch')} with {len(MATRIX.get('champions', {}))} champions.")

# Build champion catalog for picker UI
CHAMPION_CATALOG = []
for cid, champ in MATRIX.get("champions", {}).items():
    CHAMPION_CATALOG.append({
        "cid": cid,
        "name": champ["name"],
        "slug": champ["slug"],
        "roles": list(champ.get("roles", {}).keys()),
        "role_priors": champ.get("role_priors", {}),
        "icon": f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/{cid}.png"
    })
CHAMPION_CATALOG.sort(key=lambda x: x["name"])


class DraftRequestHandler(SimpleHTTPRequestHandler):
    """
    HTTP handler serving static UI assets and real-time draft recommendation API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/champions":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            payload = {
                "patch": MATRIX.get("patch", "16.19"),
                "tier": MATRIX.get("tier", "emerald_plus"),
                "champions": CHAMPION_CATALOG
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "patch": MATRIX.get("patch")}).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/score":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
                return

            assigned_role = data.get("assigned_role", "top").lower()
            locked_allies = [str(c) for c in data.get("locked_allies", []) if c]
            locked_enemies = [str(c) for c in data.get("locked_enemies", []) if c]

            # 1. Infer enemy roles
            inferred_roles = INFERENCE.infer_roles(locked_enemies, MATRIX)

            # 2. Score all candidate champions for this role
            candidates = []
            for cid in MATRIX.get("champions", {}).keys():
                # Don't recommend champions that are already picked
                if cid in locked_allies or cid in locked_enemies:
                    continue

                res = SCORER.score_candidate(
                    candidate_cid=cid,
                    assigned_role=assigned_role,
                    locked_allies=locked_allies,
                    locked_enemies=locked_enemies
                )
                if res.get("viable"):
                    res["icon"] = f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/{cid}.png"
                    candidates.append(res)

            # Sort descending by composite score
            candidates.sort(key=lambda x: x["composite_score"], reverse=True)

            # 3. Calculate team damage breakdown
            allies_damage = [SCORER.get_champion_damage_profile(cid) for cid in locked_allies]
            team_phys = sum(d["pct_physical"] for d in allies_damage) / len(allies_damage) if allies_damage else 50.0
            team_magic = sum(d["pct_magic"] for d in allies_damage) / len(allies_damage) if allies_damage else 50.0
            team_true = sum(d["pct_true"] for d in allies_damage) / len(allies_damage) if allies_damage else 0.0

            response_payload = {
                "assigned_role": assigned_role,
                "locked_allies_count": len(locked_allies),
                "locked_enemies_count": len(locked_enemies),
                "inferred_enemy_roles": inferred_roles,
                "team_damage": {
                    "pct_physical": round(team_phys, 1),
                    "pct_magic": round(team_magic, 1),
                    "pct_true": round(team_true, 1)
                },
                "total_viable_candidates": len(candidates),
                "recommendations": candidates[:15]  # Top 15 picks
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8000):
    os.makedirs(UI_DIR, exist_ok=True)
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, DraftRequestHandler)
    logger.info(f"LolDraft Simulator running at http://127.0.0.1:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)

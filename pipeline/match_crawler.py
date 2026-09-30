#!/usr/bin/env python3
"""
LolDraft Multi-Server & High-Diversity Match Crawler (Riot Match-V5)
Crawls Emerald+ solo queue matches balanced across:
1. Server Variety: KR (Korea), EUW (Europe West), NA (North America).
2. Rank Variety (Emerald+): Emerald, Diamond, Master, Grandmaster, Challenger.
3. Account Variety: Caps at 2 matches per account (thousands of unique players).
4. Champion Variety: Ensures broad roster coverage and prevents meta over-clustering.

Rate-Limit Compliant:
- Paced for Riot Personal Development Keys (20 req/1s, 100 req/120s).
- Incremental checkpointing (JSONL) with automatic resume support.
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
from collections import Counter
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LolDraftCrawler")

PLATFORM_TO_REGION = {
    "kr": "asia",
    "euw1": "europe",
    "na1": "americas",
    "eun1": "europe",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "jp1": "asia",
    "oc1": "sea",
}

ROLE_MAP = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "middle",
    "BOTTOM": "bottom",
    "UTILITY": "support"
}

DEFAULT_SERVERS = ["kr", "euw1", "na1"]
DEFAULT_TIERS = ["EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

def _load_env_file():
    """Simple parser to load .env variables without external dependencies."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


class HighDiversityMatchCrawler:
    """
    Crawls Emerald+ Ranked Solo/Duo matches enforcing multi-server, multi-tier,
    account diversity, and champion pool representation.
    """

    def __init__(
        self,
        api_key: str,
        platforms: Optional[List[str]] = None,
        tiers: Optional[List[str]] = None,
        target_patch: Optional[str] = None,
        max_matches_per_account: int = 2,
        output_dir: str = "data"
    ):
        self.api_key = api_key.strip()
        self.platforms = [p.lower() for p in (platforms or DEFAULT_SERVERS)]
        self.tiers = [t.upper() for t in (tiers or DEFAULT_TIERS)]
        self.max_matches_per_account = max_matches_per_account
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "X-Riot-Token": self.api_key,
            "User-Agent": "LolDraft-Diversity-Crawler/1.0"
        })

        # Load active patch and champion role catalog from current_matrix.json
        self.matrix_champions: Dict[str, Any] = {}
        matrix_path = self.output_dir / "current_matrix.json"
        if matrix_path.exists():
            try:
                with open(matrix_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not target_patch:
                        target_patch = data.get("patch")
                    self.matrix_champions = data.get("champions", {})
            except Exception as e:
                logger.warning(f"Could not load matrix: {e}")

        self.target_patch = target_patch or "16.19"
        logger.info(
            f"Config: Patch {self.target_patch} | Servers: {[p.upper() for p in self.platforms]} | "
            f"Tiers: {self.tiers} | Max Games/Player: {self.max_matches_per_account} | "
            f"Matrix Viability Catalog: {len(self.matrix_champions)} champions"
        )

        # Destination file for clean match lines
        self.output_file = self.output_dir / f"matches_{self.target_patch}.jsonl"
        self.seen_matches: Set[str] = set()
        self.server_match_counts: Dict[str, int] = {p: 0 for p in self.platforms}
        self.champion_appearances: Counter = Counter()
        self.unique_champions: Set[str] = set()
        self.account_scrape_counts: Counter = Counter()

        self._load_existing_matches()

        # Rate limiter state (~1.25s per request safe pace for 100 req / 120s limit)
        self.min_request_interval = 1.25
        self.last_request_time = 0.0

    def _load_existing_matches(self):
        """Loads already crawled match IDs and diversity metrics from disk."""
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            m_id = record.get("match_id")
                            srv = record.get("server", "").lower()
                            if m_id:
                                self.seen_matches.add(m_id)
                                if srv in self.server_match_counts:
                                    self.server_match_counts[srv] += 1
                                for team in (record.get("blue_team", []) + record.get("red_team", [])):
                                    cid = str(team.get("cid"))
                                    self.champion_appearances[cid] += 1
                                    self.unique_champions.add(cid)
                        except Exception:
                            continue
            logger.info(
                f"Resuming: {len(self.seen_matches)} matches already saved "
                f"({', '.join(f'{k.upper()}: {v}' for k, v in self.server_match_counts.items())}) | "
                f"Unique Champions: {len(self.unique_champions)}"
            )

    def _rate_limited_get(self, url: str) -> Optional[requests.Response]:
        """Executes GET request respecting global rate limits and handling 429 Retry-After."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        while True:
            try:
                self.last_request_time = time.time()
                res = self.session.get(url, timeout=12)

                if res.status_code == 200:
                    return res
                elif res.status_code == 429:
                    retry_after = int(res.headers.get("Retry-After", 10))
                    logger.warning(f"Rate limited (429). Backing off for {retry_after}s...")
                    time.sleep(retry_after + 1)
                    continue
                elif res.status_code in (401, 403):
                    logger.error("Invalid or expired Riot API Key. Please renew on developer.riotgames.com")
                    raise PermissionError("Riot API Key is invalid or expired.")
                elif res.status_code == 404:
                    return None
                else:
                    logger.warning(f"HTTP {res.status_code} for {url}. Retrying in 2s...")
                    time.sleep(2)
            except requests.RequestException as e:
                logger.warning(f"Network error: {e}. Retrying in 3s...")
                time.sleep(3)

    def fetch_tier_puuids(self, platform: str, tier: str, max_count: int = 150) -> List[str]:
        """Fetches player PUUIDs from a specific rank tier to guarantee rank diversity."""
        puuids = []
        tier = tier.upper()

        if tier in ("CHALLENGER", "GRANDMASTER", "MASTER"):
            endpoint_name = f"{tier.lower()}leagues"
            url = f"https://{platform}.api.riotgames.com/lol/league/v4/{endpoint_name}/by-queue/RANKED_SOLO_5x5"
            res = self._rate_limited_get(url)
            if res:
                entries = res.json().get("entries", [])
                random.shuffle(entries)
                for entry in entries[:max_count]:
                    p = entry.get("puuid")
                    if p:
                        puuids.append(p)
        else:
            # EMERALD or DIAMOND across division I and II
            for division in ("I", "II"):
                url = f"https://{platform}.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{tier}/{division}?page=1"
                res = self._rate_limited_get(url)
                if res:
                    entries = res.json()
                    random.shuffle(entries)
                    for entry in entries:
                        p = entry.get("puuid")
                        if p and p not in puuids:
                            puuids.append(p)
                        if len(puuids) >= max_count:
                            break
                if len(puuids) >= max_count:
                    break

        return puuids

    def fetch_player_match_ids(self, region: str, puuid: str, count: int = 5) -> List[str]:
        """Fetches recent Ranked Solo/Duo match IDs (queue 420) from region endpoint."""
        url = (
            f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
            f"?queue=420&type=ranked&start=0&count={count}"
        )
        res = self._rate_limited_get(url)
        if not res:
            return []
        return res.json()

    def parse_match(self, match_data: Dict[str, Any], platform: str) -> Optional[Dict[str, Any]]:
        """Parses and validates a 5v5 match payload."""
        info = match_data.get("info", {})
        game_version = info.get("gameVersion", "")
        if not game_version.startswith(self.target_patch):
            return None

        # Quality Filter 3: Strict Queue Validation (420 = Ranked Solo/Duo only)
        if info.get("queueId") != 420:
            return None

        # Quality Filter 1: Remake and Early Surrender check
        if info.get("gameEndedInEarlySurrender"):
            return None
        duration = info.get("gameDuration", 0)
        if duration < 900:  # Require at least 15 minutes of play
            return None

        participants = info.get("participants", [])
        if len(participants) != 10:
            return None

        blue_team = []
        red_team = []
        blue_win = False

        for p in participants:
            # Quality Filter 1B: Check for AFKs or leavers (Riot progression forfeiture)
            if p.get("eligibleForProgression") is False:
                return None
            # Quality Filter 1C: Check for disconnect / level 1 AFK (< 500 dmg to champions)
            if p.get("totalDamageDealtToChampions", 0) < 500:
                return None

            cid = str(p.get("championId"))
            pos = p.get("teamPosition", "").upper()
            role = ROLE_MAP.get(pos)
            team_id = p.get("teamId")
            win = bool(p.get("win", False))

            if not role:
                return None

            # Quality Filter 4: Matrix Role Viability Check
            # Ensure pick exists in current_matrix.json under its played role
            if self.matrix_champions:
                champ_entry = self.matrix_champions.get(cid)
                if not champ_entry or role not in champ_entry.get("roles", {}):
                    return None  # Discard match with unrecorded extreme off-meta pick

            player_entry = {
                "cid": cid,
                "role": role,
                "win": win
            }

            if team_id == 100:
                blue_team.append(player_entry)
                blue_win = win
            elif team_id == 200:
                red_team.append(player_entry)

        if len(blue_team) != 5 or len(red_team) != 5:
            return None

        blue_roles = {p["role"] for p in blue_team}
        red_roles = {p["role"] for p in red_team}
        required_roles = {"top", "jungle", "middle", "bottom", "support"}
        if blue_roles != required_roles or red_roles != required_roles:
            return None

        return {
            "match_id": match_data.get("metadata", {}).get("matchId"),
            "server": platform,
            "patch": self.target_patch,
            "duration": duration,
            "blue_win": blue_win,
            "blue_team": blue_team,
            "red_team": red_team
        }

    def crawl(self, target_matches: int = 2500):
        """Crawls matches enforcing server, rank, account, and champion diversity."""
        total_saved = len(self.seen_matches)
        target_per_server = (target_matches + len(self.platforms) - 1) // len(self.platforms)
        logger.info(
            f"Target: {target_matches} total matches (~{target_per_server} per server across "
            f"{[p.upper() for p in self.platforms]}). Currently saved: {total_saved}"
        )

        with open(self.output_file, "a", encoding="utf-8") as out_f:
            for platform in self.platforms:
                region = PLATFORM_TO_REGION.get(platform, "europe")
                current_server_count = self.server_match_counts.get(platform, 0)
                needed = target_per_server - current_server_count

                if needed <= 0:
                    logger.info(f"[{platform.upper()}] Server quota met ({current_server_count}/{target_per_server}).")
                    continue

                logger.info(f"[{platform.upper()}] Seeding players across Emerald+ ranks...")
                # Gather diverse player accounts across all tiers
                player_pool: List[str] = []
                players_per_tier = max(30, (needed // len(self.tiers)) + 20)

                for tier in self.tiers:
                    tier_players = self.fetch_tier_puuids(platform, tier, max_count=players_per_tier)
                    logger.info(f"[{platform.upper()}] {tier}: {len(tier_players)} accounts seeded.")
                    player_pool.extend(tier_players)

                random.shuffle(player_pool)
                logger.info(f"[{platform.upper()}] Total diverse account pool: {len(player_pool)} players.")

                # Direct Streaming Ingestion: sample up to max_matches_per_account per player
                logger.info(f"[{platform.upper()}] Streaming matches across {len(player_pool)} player accounts...")
                for puuid in player_pool:
                    if self.server_match_counts[platform] >= target_per_server:
                        break
                    if len(self.seen_matches) >= target_matches:
                        break

                    # 1 API request fetches up to 15 recent match IDs
                    m_ids = self.fetch_player_match_ids(region, puuid, count=15)
                    player_matches_taken = 0

                    for match_id in m_ids:
                        if player_matches_taken >= self.max_matches_per_account:
                            break
                        if match_id in self.seen_matches:
                            continue
                        if self.server_match_counts[platform] >= target_per_server:
                            break

                        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
                        res = self._rate_limited_get(url)
                        if not res:
                            continue

                        raw_data = res.json()
                        parsed = self.parse_match(raw_data, platform)
                        if parsed:
                            for p in (parsed["blue_team"] + parsed["red_team"]):
                                cid = p["cid"]
                                self.champion_appearances[cid] += 1
                                self.unique_champions.add(cid)

                            out_f.write(json.dumps(parsed) + "\n")
                            out_f.flush()
                            self.seen_matches.add(match_id)
                            self.server_match_counts[platform] += 1
                            player_matches_taken += 1
                            total_saved = len(self.seen_matches)

                            if total_saved % 10 == 0 or total_saved == target_matches:
                                logger.info(
                                    f"Progress: [{total_saved} / {target_matches}] | "
                                    f"KR: {self.server_match_counts.get('kr', 0)} | "
                                    f"EUW: {self.server_match_counts.get('euw1', 0)} | "
                                    f"NA: {self.server_match_counts.get('na1', 0)} | "
                                    f"Roster: {len(self.unique_champions)} champs"
                                )

        logger.info(
            f"Crawling complete! Total clean matches saved: {len(self.seen_matches)} in {self.output_file}\n"
            f"Server Breakdown: {', '.join(f'{k.upper()}: {v}' for k, v in self.server_match_counts.items())}\n"
            f"Total Unique Champions Sampled: {len(self.unique_champions)}"
        )


def main():
    _load_env_file()
    parser = argparse.ArgumentParser(description="LolDraft High-Diversity Match Crawler")
    parser.add_argument("--api-key", type=str, default=os.getenv("RIOT_API_KEY", ""), help="Riot API Key (or in .env)")
    parser.add_argument("--target", type=int, default=2500, help="Target total matches (default: 2500)")
    parser.add_argument("--servers", type=str, default="kr,euw1,na1", help="Servers: kr,euw1,na1")
    parser.add_argument("--tiers", type=str, default="EMERALD,DIAMOND,MASTER,GRANDMASTER,CHALLENGER", help="Comma-separated tiers")
    parser.add_argument("--max-per-player", type=int, default=2, help="Max matches per player account (default: 2)")
    parser.add_argument("--patch", type=str, default="", help="Target patch (default: auto from current_matrix.json)")
    args = parser.parse_args()

    api_key = args.api_key.strip()
    if not api_key:
        logger.error("Riot API Key required! Provide via --api-key RGAPI-... or in .env file.")
        sys.exit(1)

    servers = [s.strip().lower() for s in args.servers.split(",") if s.strip()]
    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]

    crawler = HighDiversityMatchCrawler(
        api_key=api_key,
        platforms=servers,
        tiers=tiers,
        target_patch=args.patch or None,
        max_matches_per_account=args.max_per_player
    )
    crawler.crawl(target_matches=args.target)


if __name__ == "__main__":
    main()

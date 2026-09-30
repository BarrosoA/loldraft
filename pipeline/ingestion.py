#!/usr/bin/env python3
"""
LolDraft Ingestion Pipeline
Scrapes champion statistics, lane counters, and team synergies from Lolalytics (emerald_plus).
Supports multi-lane extraction for flex champions, empirical role priors, and persistent disk caching.
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LolDraftIngestion")

class LolalyticsScraper:
    """
    Robust scraper for Lolalytics pre-aggregated statistics.
    Extracts champion baseline win rates, 5-role enemy counter matchups, 4-role team synergies,
    and empirical role priors across all viable lanes.
    """

    BASE_URL = "https://lolalytics.com"
    MEGA_URL = "https://a1.lolalytics.com/mega/"
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://lolalytics.com/",
    }

    def __init__(
        self,
        cache_dir: str = "data/cache",
        delay_seconds: float = 0.35,
        lane_threshold: float = 5.0,
        max_retries: int = 3,
        timeout: int = 15
    ):
        self.cache_dir = cache_dir
        self.delay_seconds = delay_seconds
        self.lane_threshold = lane_threshold
        self.timeout = timeout
        os.makedirs(self.cache_dir, exist_ok=True)

        # Setup resilient HTTP session with exponential backoff retries
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(self.DEFAULT_HEADERS)

    def _deep_resolve(self, val: Any, objs: List[Any]) -> Any:
        """
        Recursively resolves serialized base-36 object indices from Qwik state payloads.
        """
        if isinstance(val, str):
            try:
                idx = int(val, 36)
                if 0 <= idx < len(objs):
                    return self._deep_resolve(objs[idx], objs)
            except Exception:
                pass
        elif isinstance(val, list):
            return [self._deep_resolve(x, objs) for x in val]
        elif isinstance(val, dict):
            return {k: self._deep_resolve(v, objs) for k, v in val.items()}
        return val

    def get_meta_info(self, tier: str = "emerald_plus") -> Dict[str, Any]:
        """
        Queries Lolalytics front-end endpoint to get active patches and average tier win rate.
        """
        url = f"{self.MEGA_URL}?ep=front&v=1&tier={tier}&queue=ranked&region=all"
        logger.info(f"Querying meta info from {url}")
        res = self.session.get(url, timeout=self.timeout)
        if res.status_code != 200:
            raise RuntimeError(f"Failed to fetch meta info: HTTP {res.status_code}")
        
        data = res.json()
        patches = data.get("patches", [])
        active_patch = patches[0] if patches else "16.19"
        avg_wr = data.get("avgWr", 51.5)
        logger.info(f"Discovered active patch: {active_patch}, Emerald+ Avg WR: {avg_wr}%")
        return {
            "patch": active_patch,
            "available_patches": patches,
            "avg_wr": avg_wr,
            "total_analysed": data.get("current", {}).get("analysed", 0)
        }

    def get_champion_list(self) -> Dict[str, int]:
        """
        Fetches the complete mapping of champion slugs to champion IDs from the tier list page.
        """
        url = f"{self.BASE_URL}/lol/tierlist/"
        logger.info("Fetching complete champion catalog from tier list...")
        res = self.session.get(url, timeout=self.timeout)
        if res.status_code != 200:
            raise RuntimeError(f"Failed to fetch tierlist: HTTP {res.status_code}")
        
        match = re.search(r'<script type="qwik/json">([\s\S]*?)</script>', res.text)
        if not match:
            raise ValueError("Unable to extract qwik/json block from tierlist page")
        
        qwik_data = json.loads(match.group(1))
        objs = qwik_data.get("objs", [])
        
        champ_id_map = {}
        for obj in objs:
            if isinstance(obj, dict) and "champId" in obj:
                resolved = self._deep_resolve(obj["champId"], objs)
                if isinstance(resolved, dict):
                    champ_id_map = {k: int(v) for k, v in resolved.items() if str(v).isdigit()}
                    break

        if not champ_id_map:
            raise ValueError("Could not locate champId dictionary in serialized objects")
            
        logger.info(f"Loaded {len(champ_id_map)} playable champions.")
        return champ_id_map

    def _parse_page_qwik(self, html: str, champ_slug: str) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any], Dict[str, Any]]:
        """
        Extracts champion metadata, lane distribution, stats, and enemy counters from HTML.
        """
        match = re.search(r'<script type="qwik/json">([\s\S]*?)</script>', html)
        if not match:
            return {}, {}, {}, {}
        try:
            qwik_data = json.loads(match.group(1))
            objs = qwik_data.get("objs", [])
        except Exception as e:
            logger.error(f"Error parsing JSON block for {champ_slug}: {e}")
            return {}, {}, {}, {}

        champ_info = {}
        stats_info = {}
        enemy_data = {}
        lanes_dict = {}

        for obj in objs:
            if isinstance(obj, dict):
                if "cid" in obj and "champName" in obj and "patch" in obj:
                    champ_info = {
                        "cid": self._deep_resolve(obj.get("cid"), objs),
                        "slug": champ_slug,
                        "name": "Vi" if champ_slug.lower() == "vi" else self._deep_resolve(obj.get("champName"), objs),
                        "patch": self._deep_resolve(obj.get("patch"), objs),
                        "tier": self._deep_resolve(obj.get("tier"), objs),
                    }
                if "wr" in obj and "avgWr" in obj and "n" in obj and "damage" in obj:
                    stats_info = {
                        "default_lane": self._deep_resolve(obj.get("defaultLane"), objs),
                        "lane": self._deep_resolve(obj.get("lane"), objs),
                        "win_rate": self._deep_resolve(obj.get("wr"), objs),
                        "avg_win_rate": self._deep_resolve(obj.get("avgWr"), objs),
                        "avg_wr_delta": self._deep_resolve(obj.get("avgWrDelta"), objs),
                        "pick_rate": self._deep_resolve(obj.get("pr"), objs),
                        "ban_rate": self._deep_resolve(obj.get("br"), objs),
                        "games": self._deep_resolve(obj.get("n"), objs),
                        "damage": self._deep_resolve(obj.get("damage"), objs),
                    }
                if "enemy" in obj:
                    resolved_enemy = self._deep_resolve(obj["enemy"], objs)
                    if isinstance(resolved_enemy, dict) and "top" in resolved_enemy:
                        enemy_data = resolved_enemy
                if "lanes" in obj:
                    resolved_lanes = self._deep_resolve(obj["lanes"], objs)
                    if isinstance(resolved_lanes, dict):
                        lanes_dict = {
                            k: float(v) for k, v in resolved_lanes.items()
                            if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())
                        }

        return champ_info, lanes_dict, stats_info, enemy_data

    def _fetch_team_synergies(self, champ_slug: str, patch: str, lane: str, tier: str = "emerald_plus") -> Dict[str, Any]:
        """
        Fetches ally co-occurrence stats for a specific champion and lane.
        """
        team_url = (
            f"{self.MEGA_URL}?ep=build-team&v=1&patch={patch}"
            f"&c={champ_slug}&lane={lane}&tier={tier}&queue=ranked&region=all"
        )
        time.sleep(self.delay_seconds)
        try:
            res_team = self.session.get(team_url, timeout=self.timeout)
            if res_team.status_code == 200:
                team_json = res_team.json()
                return team_json.get("team", {})
        except Exception as e:
            logger.warning(f"Could not fetch team synergy JSON for {champ_slug} ({lane}): {e}")
        return {}

    def scrape_champion(
        self,
        champ_slug: str,
        patch: str,
        tier: str = "emerald_plus",
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Scrapes a champion's datasets across all viable roles (>= lane_threshold % pick rate).
        Caches consolidated payload with role_priors and nested roles to disk.
        """
        patch_dir = os.path.join(self.cache_dir, patch)
        os.makedirs(patch_dir, exist_ok=True)
        cache_path = os.path.join(patch_dir, f"{champ_slug}.json")

        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if "roles" in cached_data and "role_priors" in cached_data:
                        return cached_data
            except Exception as e:
                logger.warning(f"Corrupt or legacy cache file for {champ_slug}, re-fetching: {e}")

        # Step 1: Fetch primary build page
        page_url = f"{self.BASE_URL}/lol/{champ_slug}/build/?tier={tier}&patch={patch}"
        time.sleep(self.delay_seconds)
        res = self.session.get(page_url, timeout=self.timeout)
        if res.status_code != 200:
            logger.error(f"HTTP {res.status_code} fetching page for {champ_slug}")
            return None

        champ_info, lanes_dict, primary_stats, primary_enemy = self._parse_page_qwik(res.text, champ_slug)
        if not champ_info or not primary_stats:
            logger.error(f"Failed to parse build page for {champ_slug}")
            return None

        primary_lane = primary_stats.get("lane") or primary_stats.get("default_lane") or "middle"
        primary_synergies = self._fetch_team_synergies(champ_slug, patch, primary_lane, tier)

        roles = {
            primary_lane: {
                "stats": primary_stats,
                "enemy_counters": primary_enemy,
                "ally_synergies": primary_synergies
            }
        }

        # Calculate empirical role priors and identify viable secondary lanes
        role_priors = {}
        viable_secondary = []
        if lanes_dict:
            total_pct = sum(lanes_dict.values()) or 100.0
            role_priors = {k: round(v / total_pct, 4) for k, v in lanes_dict.items() if v > 0}
            viable_secondary = [
                l for l, pct in lanes_dict.items()
                if pct >= self.lane_threshold and l != primary_lane
            ]
        else:
            role_priors = {primary_lane: 1.0}

        # Step 2: Ingest viable secondary lanes
        for sec_lane in viable_secondary:
            pct_val = lanes_dict.get(sec_lane, 0)
            logger.info(f"  -> Ingesting secondary lane '{sec_lane}' ({pct_val:.1f}% pick rate) for {champ_slug}...")
            sec_url = f"{self.BASE_URL}/lol/{champ_slug}/build/?tier={tier}&patch={patch}&lane={sec_lane}"
            time.sleep(self.delay_seconds)
            res_sec = self.session.get(sec_url, timeout=self.timeout)
            if res_sec.status_code == 200:
                _, _, sec_stats, sec_enemy = self._parse_page_qwik(res_sec.text, champ_slug)
                if sec_stats and sec_enemy:
                    sec_synergies = self._fetch_team_synergies(champ_slug, patch, sec_lane, tier)
                    roles[sec_lane] = {
                        "stats": sec_stats,
                        "enemy_counters": sec_enemy,
                        "ally_synergies": sec_synergies
                    }
                else:
                    logger.warning(f"Could not parse secondary lane {sec_lane} for {champ_slug}")
            else:
                logger.warning(f"HTTP {res_sec.status_code} fetching secondary lane {sec_lane} for {champ_slug}")

        payload = {
            "info": champ_info,
            "role_priors": role_priors,
            "roles": roles
        }

        # Cache payload to disk
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache data for {champ_slug}: {e}")

        return payload

    def scrape_all(
        self,
        patch: Optional[str] = None,
        tier: str = "emerald_plus",
        limit: Optional[int] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Iterates across all champions in the game, fetching their complete dataset.
        """
        meta = self.get_meta_info(tier=tier)
        target_patch = patch or meta["patch"]
        avg_wr = meta["avg_wr"]

        champion_catalog = self.get_champion_list()
        champ_slugs = sorted(list(champion_catalog.keys()))

        if limit:
            logger.info(f"Limiting scrape to first {limit} champions as requested.")
            champ_slugs = champ_slugs[:limit]

        total = len(champ_slugs)
        logger.info(f"Starting ingestion for {total} champions on Patch {target_patch} ({tier})...")

        results = {}
        success_count = 0
        cached_count = 0

        for i, slug in enumerate(champ_slugs, 1):
            cache_file = os.path.join(self.cache_dir, target_patch, f"{slug}.json")
            was_cached = os.path.exists(cache_file) and not force_refresh

            logger.info(f"[{i}/{total}] Ingesting {slug.upper()} ({'CACHED' if was_cached else 'FETCHING'})...")
            data = self.scrape_champion(
                champ_slug=slug,
                patch=target_patch,
                tier=tier,
                force_refresh=force_refresh
            )

            if data and data.get("roles"):
                results[slug] = data
                success_count += 1
                if was_cached:
                    cached_count += 1
            else:
                logger.warning(f"Failed to ingest valid data for {slug}")

        logger.info(
            f"Ingestion complete: {success_count}/{total} champions ingested "
            f"({cached_count} from cache, {success_count - cached_count} network requests)."
        )

        return {
            "patch": target_patch,
            "tier": tier,
            "avg_tier_wr": avg_wr,
            "champions": results
        }


def main():
    parser = argparse.ArgumentParser(description="LolDraft Statistics Ingestion Tool")
    parser.add_argument("--patch", type=str, default=None, help="Target patch version (e.g., 16.19)")
    parser.add_argument("--tier", type=str, default="emerald_plus", help="Target rank tier (default: emerald_plus)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of champions to scrape (for testing)")
    parser.add_argument("--delay", type=float, default=0.35, help="Delay between HTTP requests in seconds (default: 0.35)")
    parser.add_argument("--lane-threshold", type=float, default=5.0, help="Minimum pick rate percentage to scrape a role (default: 5.0)")
    parser.add_argument("--force", action="store_true", help="Force refresh existing disk cache")
    parser.add_argument("--build-matrix", action="store_true", help="Automatically compile into current_matrix.json after scraping")

    args = parser.parse_args()

    scraper = LolalyticsScraper(
        cache_dir="data/cache",
        delay_seconds=args.delay,
        lane_threshold=args.lane_threshold
    )

    scraped_data = scraper.scrape_all(
        patch=args.patch,
        tier=args.tier,
        limit=args.limit,
        force_refresh=args.force
    )

    if args.build_matrix:
        try:
            from pipeline.matrix_builder import MatrixBuilder
        except ImportError:
            from matrix_builder import MatrixBuilder
        builder = MatrixBuilder()
        builder.build_matrix_from_scraped_data(scraped_data)


if __name__ == "__main__":
    main()

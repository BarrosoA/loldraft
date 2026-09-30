#!/usr/bin/env python3
"""
LolDraft LCU WebSocket Listener
Connects to the active League Client via authenticated WebSocket (WSS) with self-signed TLS.
Subscribes to OnJsonApiEvent_lol-champ-select_v1_session and emits deduplicated live draft states.
"""

import sys
import os
import ssl
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
import aiohttp

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connector.lockfile import LCULockfileDetector, LCUCredentials
from pipeline.role_inference import DraftScorer

logger = logging.getLogger("LolDraftSocket")

# Mapping of Riot assignedPosition to standard role strings
POSITION_MAP = {
    "top": "top",
    "jungle": "jungle",
    "middle": "middle",
    "mid": "middle",
    "bottom": "bottom",
    "bot": "bottom",
    "utility": "support",
    "support": "support",
    "": "top"  # fallback
}

class LCUSocketListener:
    """
    Subscribes to live LCU champion select events over WebSocket.
    """

    def __init__(
        self,
        scorer: Optional[DraftScorer] = None,
        on_update_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.scorer = scorer
        self.on_update_callback = on_update_callback
        self._last_state_hash = None
        self._is_running = False

    def _parse_draft_state(self, session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extracts clean, structured draft state from raw Riot session payload.
        """
        if not session_data or not isinstance(session_data, dict):
            return None

        local_cell_id = session_data.get("localPlayerCellId", -1)
        my_team = session_data.get("myTeam", [])
        their_team = session_data.get("theirTeam", [])
        timer = session_data.get("timer", {})
        actions = session_data.get("actions", [])

        # Find local player's role
        local_role = "top"
        local_picked = False
        locked_allies = []

        for player in my_team:
            cell_id = player.get("cellId")
            cid = player.get("championId", 0)
            pos = player.get("assignedPosition", "").lower()
            std_role = POSITION_MAP.get(pos, "top")

            if cell_id == local_cell_id:
                local_role = std_role
                if cid > 0:
                    local_picked = True
            else:
                if cid > 0:
                    locked_allies.append(str(cid))

        # Locked enemies
        locked_enemies = []
        for player in their_team:
            cid = player.get("championId", 0)
            if cid > 0:
                locked_enemies.append(str(cid))

        # Determine if it is currently the local player's turn to pick
        is_my_turn = False
        for action_group in actions:
            for act in action_group:
                if (
                    act.get("actorCellId") == local_cell_id
                    and act.get("isInProgress")
                    and not act.get("completed")
                    and act.get("type") == "pick"
                ):
                    is_my_turn = True
                    break

        return {
            "phase": timer.get("phase", "UNKNOWN"),
            "assigned_role": local_role,
            "local_picked": local_picked,
            "is_my_turn": is_my_turn,
            "locked_allies": sorted(locked_allies),
            "locked_enemies": sorted(locked_enemies),
            "time_left": round(timer.get("adjustedTimeLeftInPhase", 0) / 1000.0, 1)
        }

    async def connect_and_listen(self, creds: Optional[LCUCredentials] = None):
        """
        Establishes WSS connection and processes events until disconnect.
        """
        if not creds:
            creds = LCULockfileDetector.find_credentials()
            if not creds:
                logger.error("League Client is not running. Unable to connect.")
                return

        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        headers = {
            "Authorization": creds.auth_header
        }

        logger.info(f"Connecting to League WebSocket at {creds.ws_url}...")
        self._is_running = True

        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(
                    creds.ws_url,
                    headers=headers,
                    ssl=ssl_ctx
                ) as ws:
                    logger.info("Connected to League Client. Subscribing to Champ Select events...")
                    # Subscribe to champion select session topic (Wamp event #5)
                    await ws.send_str('[5, "OnJsonApiEvent_lol-champ-select_v1_session"]')

                    async for msg in ws:
                        if not self._is_running:
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                payload = json.loads(msg.data)
                                # WAMP format: [opcode, topic, data]
                                if isinstance(payload, list) and len(payload) >= 3:
                                    event_data = payload[2]
                                    session_data = event_data.get("data", {})
                                    parsed = self._parse_draft_state(session_data)

                                    if parsed:
                                        # Deduplicate heartbeat ticks (only trigger on lock-in or turn change)
                                        state_hash = (
                                            parsed["assigned_role"],
                                            tuple(parsed["locked_allies"]),
                                            tuple(parsed["locked_enemies"]),
                                            parsed["is_my_turn"],
                                            parsed["phase"]
                                        )

                                        if state_hash != self._last_state_hash:
                                            self._last_state_hash = state_hash
                                            self._handle_draft_update(parsed)

                            except Exception as e:
                                logger.debug(f"Error handling message: {e}")

                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.info("WebSocket connection closed.")
                            break

            except Exception as e:
                logger.error(f"WebSocket error: {e}")

    def _handle_draft_update(self, draft_state: Dict[str, Any]):
        """
        Dispatches updated draft state and computes live recommendations.
        """
        role = draft_state["assigned_role"]
        allies = draft_state["locked_allies"]
        enemies = draft_state["locked_enemies"]
        turn = ">>> YOUR TURN TO PICK! <<<" if draft_state["is_my_turn"] else "Waiting for players..."

        print("\n" + "="*60)
        print(f"LIVE CHAMP SELECT EVENT • Assigned Role: {role.upper()}")
        print(f"Status: {turn}")
        print(f"Locked Allies ({len(allies)}): {allies}")
        print(f"Locked Enemies ({len(enemies)}): {enemies}")
        print("="*60)

        if self.scorer:
            ranked = []
            for cid in self.scorer.matrix.get("champions", {}).keys():
                if cid in allies or cid in enemies:
                    continue
                res = self.scorer.score_candidate(cid, role, allies, enemies)
                if res.get("viable"):
                    ranked.append(res)

            ranked.sort(key=lambda x: x["composite_score"], reverse=True)

            print(f"\nTOP 5 PICKS FOR {role.upper()}:")
            for i, c in enumerate(ranked[:5], 1):
                lane_s = f"+{c['expected_lane_delta']}%" if c['expected_lane_delta'] >= 0 else f"{c['expected_lane_delta']}%"
                print(f"  #{i} {c['name']:<12} | Score: {c['composite_score']}% | Lane: {lane_s} | Rationale: {c['rationale']}")
            print("")

        if self.on_update_callback:
            self.on_update_callback(draft_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load matrix for live scoring
    matrix_file = "data/current_matrix.json"
    scorer = None
    if os.path.exists(matrix_file):
        with open(matrix_file, "r", encoding="utf-8") as f:
            matrix = json.load(f)
            scorer = DraftScorer(matrix)

    listener = LCUSocketListener(scorer=scorer)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(listener.connect_and_listen())
    except KeyboardInterrupt:
        print("\nDisconnected.")

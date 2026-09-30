#!/usr/bin/env python3
"""
LolDraft LCU Client Detector & Lockfile Parser
Locates the running League of Legends client process and extracts authentication credentials.
Supports direct process memory inspection via psutil and disk lockfile discovery.
"""

import os
import re
import time
import base64
import logging
from dataclasses import dataclass
from typing import Optional, List
import psutil

logger = logging.getLogger("LolDraftLockfile")

COMMON_INSTALL_PATHS = [
    r"C:\Riot Games\League of Legends",
    r"D:\Riot Games\League of Legends",
    r"E:\Riot Games\League of Legends",
    r"C:\Program Files\Riot Games\League of Legends",
    r"C:\Program Files (x86)\Riot Games\League of Legends",
]

@dataclass
class LCUCredentials:
    """Represents local client credentials needed to authenticate with Riot's LCU API."""
    port: int
    password: str
    protocol: str = "https"
    pid: Optional[int] = None
    install_dir: Optional[str] = None

    @property
    def auth_header(self) -> str:
        """Returns standard HTTP Basic Authentication header for Riot LCU."""
        token = base64.b64encode(f"riot:{self.password}".encode("ascii")).decode("ascii")
        return f"Basic {token}"

    @property
    def rest_url(self) -> str:
        """Returns base URL for REST requests."""
        return f"{self.protocol}://127.0.0.1:{self.port}"

    @property
    def ws_url(self) -> str:
        """Returns base URL for WebSocket connections."""
        return f"wss://127.0.0.1:{self.port}"


class LCULockfileDetector:
    """
    Detects running LeagueClientUx instances and parses authentication parameters.
    """

    @classmethod
    def get_credentials_from_process(cls) -> Optional[LCUCredentials]:
        """
        Inspects running processes for LeagueClientUx.exe and extracts command-line flags.
        """
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
                pname = proc.info.get("name") or ""
                if pname.lower() in ("leagueclientux.exe", "leagueclient.exe"):
                    cmdline = proc.info.get("cmdline") or []
                    cmd_str = " ".join(cmdline)

                    port_match = re.search(r"--app-port=(\d+)", cmd_str)
                    auth_match = re.search(r"--remoting-auth-token=([a-zA-Z0-9_-]+)", cmd_str)

                    if port_match and auth_match:
                        port = int(port_match.group(1))
                        password = auth_match.group(1)
                        install_dir = proc.info.get("cwd") or None
                        logger.info(f"Found active League Client process (PID {proc.pid}) on port {port}.")
                        return LCUCredentials(
                            port=port,
                            password=password,
                            pid=proc.pid,
                            install_dir=install_dir
                        )
        except Exception as e:
            logger.debug(f"Error inspecting processes: {e}")
        return None

    @classmethod
    def parse_lockfile_content(cls, content: str) -> Optional[LCUCredentials]:
        """
        Parses the raw 5-tuple lockfile format:
        ProcessName:PID:Port:Password:Protocol
        """
        parts = content.strip().split(":")
        if len(parts) >= 5:
            try:
                pid = int(parts[1])
                port = int(parts[2])
                password = parts[3]
                protocol = parts[4]
                return LCUCredentials(
                    port=port,
                    password=password,
                    protocol=protocol,
                    pid=pid
                )
            except ValueError:
                pass
        return None

    @classmethod
    def get_credentials_from_disk(cls) -> Optional[LCUCredentials]:
        """
        Checks common installation directories for an active lockfile.
        """
        for path in COMMON_INSTALL_PATHS:
            lockfile_path = os.path.join(path, "lockfile")
            if os.path.exists(lockfile_path):
                try:
                    with open(lockfile_path, "r", encoding="utf-8") as f:
                        creds = cls.parse_lockfile_content(f.read())
                        if creds:
                            creds.install_dir = path
                            logger.info(f"Loaded credentials from lockfile at {lockfile_path}.")
                            return creds
                except Exception as e:
                    logger.debug(f"Could not read lockfile at {lockfile_path}: {e}")
        return None

    @classmethod
    def find_credentials(cls) -> Optional[LCUCredentials]:
        """
        Attempts to resolve credentials via running process first, then falls back to disk lockfile.
        """
        creds = cls.get_credentials_from_process()
        if creds:
            return creds
        return cls.get_credentials_from_disk()

    @classmethod
    def wait_for_client(
        cls,
        timeout_seconds: Optional[float] = None,
        poll_interval: float = 1.5
    ) -> Optional[LCUCredentials]:
        """
        Blocks and polls until the League Client is detected or timeout expires.
        """
        start = time.time()
        logger.info("Waiting for League Client to launch...")
        while True:
            creds = cls.find_credentials()
            if creds:
                return creds
            if timeout_seconds and (time.time() - start) >= timeout_seconds:
                logger.info(f"Timed out after {timeout_seconds}s waiting for League Client.")
                return None
            time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    creds = LCULockfileDetector.find_credentials()
    if creds:
        print("\n=== Active League Client Detected ===")
        print(f"Port:     {creds.port}")
        print(f"PID:      {creds.pid}")
        print(f"REST URL: {creds.rest_url}")
        print(f"WS URL:   {creds.ws_url}")
        print(f"Auth:     {creds.auth_header[:15]}...")
    else:
        print("\nLeague Client is not currently running.")

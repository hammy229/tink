"""Profiles: each person Tink speaks for gets their own sounds, calibration, and conversation state.

Local mode (default): one profile, "local", stored directly in TINK_DATA, exactly as before profiles.
Public mode (TINK_PUBLIC=1): every visitor creates a profile. Its id is a random, unguessable token that
works like a private link, and limits keep a shared server healthy.

Layout in public mode:
    data/profiles/<id>/profile.json   {"name": "Maya", "created": "2026-10-24T16:05"}
    data/profiles/<id>/sounds.json, usage.json, takes/...   (the same layout Store uses)
"""
import asyncio
import json
import os
import re
import secrets
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .store import DATA, Store
from .turn import Pending

PUBLIC = os.environ.get("TINK_PUBLIC") == "1"
LOCAL = "local"
MAX_PROFILES = int(os.environ.get("TINK_MAX_PROFILES", "500"))
MAX_LOADED = 32  # profiles kept in memory; older ones reload from disk on their next request
MAX_SOUNDS = 16  # public mode only
MAX_TAKES = 12  # per sound, public mode only
ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,32}$")


@dataclass
class Profile:
    id: str
    name: str
    store: Store
    recent: deque = field(default_factory=lambda: deque(maxlen=5))  # what Tink said lately, for the LLM
    pending: Pending | None = None  # an open yes/no question
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # one turn at a time, per person


class Profiles:
    def __init__(self, root: Path = DATA):
        self.root = root
        self.loaded: OrderedDict[str, Profile] = OrderedDict()
        self._lock = threading.RLock()  # requests load profiles from worker threads; two Stores would lose writes

    def _dir(self, pid: str) -> Path:
        return self.root if pid == LOCAL else self.root / "profiles" / pid

    def count(self) -> int:
        d = self.root / "profiles"
        return sum(1 for p in d.iterdir() if (p / "profile.json").exists()) if d.exists() else 0

    def create(self, name: str) -> Profile:
        with self._lock:
            return self._create(name)

    def _create(self, name: str) -> Profile:
        if self.count() >= MAX_PROFILES:
            raise OverflowError("This Tink server is full. Try again later or run Tink on your own laptop.")
        pid = secrets.token_urlsafe(12)
        d = self._dir(pid)
        d.mkdir(parents=True)
        created = datetime.now().isoformat(timespec="minutes")
        (d / "profile.json").write_text(json.dumps({"name": name, "created": created}))
        return self.get(pid)

    def get(self, pid: str | None) -> Profile:
        """The profile for an id. No id means the local profile, which public mode does not have."""
        with self._lock:
            return self._get(pid)

    def _get(self, pid: str | None) -> Profile:
        if not pid:
            if PUBLIC:
                raise KeyError("Create a profile first.")
            pid = LOCAL
        if pid != LOCAL and not ID_RE.match(pid):
            raise KeyError("That profile link is not valid.")
        if pid in self.loaded:
            self.loaded.move_to_end(pid)
            return self.loaded[pid]
        d = self._dir(pid)
        if pid == LOCAL:
            if PUBLIC:
                raise KeyError("Create a profile first.")
            name = "This laptop"
        elif (d / "profile.json").exists():
            name = json.loads((d / "profile.json").read_text()).get("name", "Tink")
        else:
            raise KeyError("No profile with that link. It may have been removed.")
        profile = Profile(pid, name, Store(d))
        self.loaded[pid] = profile
        while len(self.loaded) > MAX_LOADED:
            self.loaded.popitem(last=False)
        return profile

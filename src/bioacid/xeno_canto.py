"""Xeno-canto API client and pseudo-label heuristics.

`Xeno-canto <https://xeno-canto.org>`_ exposes a public REST API (v3) that
returns audio recordings with metadata. This module wraps the search +
download endpoints used by M4 to acquire neotropical training data, and
implements the recordist+date+locality pseudo-label heuristic that replaces
the spatial pseudo-labels used by Lapp et al.

API base: https://xeno-canto.org/api/3/recordings

Quality filters used:
- ``q:A`` and ``q:B`` recordings only (the two best tiers).
- Country filter via ``cnt:`` keyword.
- Minimum duration via ``len_gt:`` keyword.

Pseudo-label rule:
- Two recordings are treated as the **same individual** if they share
  ``recordist`` *and* were made within ``time_window_days`` *and* their
  ``locality`` strings match exactly (or coordinates fall within
  ``location_radius_m``).
- This is a heuristic with false-positives (one recordist may record
  multiple birds at the same site on the same day) and false-negatives
  (the same bird re-located by a different recordist a year later). The
  trade-off is documented in ``reports/neotropical_extension.md``.

Network is required. The sandbox used for the M0-M3 development does not
allow connections to ``xeno-canto.org``, so this module is exercised offline
via unit tests in ``tests/test_xeno_canto.py`` and is expected to run
end-to-end on the user's own machine.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_BASE = "https://xeno-canto.org/api/3/recordings"
USER_AGENT = "bioacid/0.0.1 (+https://github.com/Lynnbrosa/biotuts)"


@dataclass(frozen=True)
class XCRecording:
    """One Xeno-canto recording entry (subset of fields actually used)."""

    id: str
    species: str
    recordist: str
    country: str
    locality: str
    latitude: float | None
    longitude: float | None
    date: str
    file_url: str
    quality: str
    length_s: float

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> XCRecording:
        try:
            length_str = payload.get("length", "0:00")
            minutes, seconds = (int(x) for x in length_str.split(":")[:2])
            length_s: float = float(minutes * 60 + seconds)
        except (ValueError, AttributeError):
            length_s = 0.0
        return cls(
            id=str(payload["id"]),
            species=f"{payload.get('gen', '')} {payload.get('sp', '')}".strip(),
            recordist=str(payload.get("rec", "")),
            country=str(payload.get("cnt", "")),
            locality=str(payload.get("loc", "")),
            latitude=_safe_float(payload.get("lat")),
            longitude=_safe_float(payload.get("lng")),
            date=str(payload.get("date", "")),
            file_url=str(payload.get("file", "")),
            quality=str(payload.get("q", "")),
            length_s=float(length_s),
        )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class SearchQuery:
    """Build a Xeno-canto query string."""

    species: str  # e.g. "Turdus rufiventris"
    country: str | None = "brazil"
    qualities: tuple[str, ...] = ("A", "B")
    min_length_s: int = 5

    def as_string(self) -> str:
        parts = [self.species]
        if self.country:
            parts.append(f"cnt:{self.country}")
        if self.min_length_s:
            parts.append(f"len_gt:{self.min_length_s}")
        if self.qualities:
            joined_q = " OR ".join(f"q:{q}" for q in self.qualities)
            parts.append(f"({joined_q})")
        return " ".join(parts)


def search(query: SearchQuery, *, max_pages: int = 10) -> list[XCRecording]:
    """Fetch all matching recordings from Xeno-canto, paginated."""
    out: list[XCRecording] = []
    for page in range(1, max_pages + 1):
        params = urllib.parse.urlencode({"query": query.as_string(), "page": page})
        url = f"{API_BASE}?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        recordings = payload.get("recordings", [])
        if not recordings:
            break
        out.extend(XCRecording.from_api(r) for r in recordings)
        if page >= int(payload.get("numPages", page)):
            break
        time.sleep(0.2)  # be polite
    return out


def download(recordings: Iterable[XCRecording], *, target_dir: Path) -> list[Path]:
    """Download audio files into ``target_dir``. Skips existing files."""
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for rec in recordings:
        if not rec.file_url:
            continue
        dest = target_dir / f"{rec.id}.mp3"
        if dest.exists():
            paths.append(dest)
            continue
        request = urllib.request.Request(rec.file_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as resp:
            dest.write_bytes(resp.read())
        paths.append(dest)
        time.sleep(0.5)
    return paths


@dataclass
class PseudoLabelConfig:
    """Knobs for the recordist+date+locality pseudo-label heuristic."""

    time_window_days: int = 1
    location_radius_m: float = 500.0


@dataclass
class PseudoLabel:
    """A pseudo-individual identifier with provenance for traceability."""

    individual_id: int
    recordist: str
    locality: str
    representative_date: str
    member_ids: tuple[str, ...]
    member_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_count", len(self.member_ids))


def assign_pseudo_labels(
    recordings: Sequence[XCRecording], *, config: PseudoLabelConfig | None = None
) -> dict[str, PseudoLabel]:
    """Group recordings into pseudo-individuals via recordist+date+locality.

    Two recordings collapse into the same pseudo-individual when:
      1. ``recordist`` strings match exactly, and
      2. dates differ by at most ``time_window_days``, and
      3. localities are identical *or* coordinates are within
         ``location_radius_m``.

    Returns a mapping ``recording_id -> PseudoLabel``. Pseudo-individuals
    are numbered sequentially starting at 0.
    """
    cfg = config or PseudoLabelConfig()
    clusters: list[list[XCRecording]] = []
    for rec in recordings:
        placed = False
        for cluster in clusters:
            anchor = cluster[0]
            if _same_individual(anchor, rec, cfg):
                cluster.append(rec)
                placed = True
                break
        if not placed:
            clusters.append([rec])

    labels: dict[str, PseudoLabel] = {}
    for idx, cluster in enumerate(clusters):
        anchor = cluster[0]
        ids = tuple(r.id for r in cluster)
        label = PseudoLabel(
            individual_id=idx,
            recordist=anchor.recordist,
            locality=anchor.locality,
            representative_date=anchor.date,
            member_ids=ids,
        )
        for rec_id in ids:
            labels[rec_id] = label
    return labels


def _same_individual(a: XCRecording, b: XCRecording, cfg: PseudoLabelConfig) -> bool:
    if a.recordist != b.recordist or not a.recordist:
        return False
    if not _within_time_window(a.date, b.date, cfg.time_window_days):
        return False
    if a.locality and a.locality == b.locality:
        return True
    if (
        a.latitude is not None
        and a.longitude is not None
        and b.latitude is not None
        and b.longitude is not None
    ):
        return (
            _haversine_m(a.latitude, a.longitude, b.latitude, b.longitude) <= cfg.location_radius_m
        )
    return False


def _within_time_window(date_a: str, date_b: str, window_days: int) -> bool:
    from datetime import date

    try:
        a = date.fromisoformat(date_a)
        b = date.fromisoformat(date_b)
    except ValueError:
        return date_a == date_b
    return abs((a - b).days) <= window_days


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in meters."""
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_m * c


__all__ = [
    "API_BASE",
    "PseudoLabel",
    "PseudoLabelConfig",
    "SearchQuery",
    "XCRecording",
    "assign_pseudo_labels",
    "download",
    "search",
]

"""M4: Pull recordings of a target species from Xeno-canto.

Defaults to sabiá-laranjeira (*Turdus rufiventris*), the M4 candidate
identified in ``CLAUDE.md``: Brazilian, stereotyped song, abundant on
Xeno-canto, present in both urban and forest habitats.

Downloads quality A/B recordings from Brazil with length >= 5s, writes
metadata + audio to ``data/raw/xenocanto/<species>/``, and assigns the
recordist+date+locality pseudo-labels described in
``reports/neotropical_extension.md``.

**Network requirement**: this script calls ``xeno-canto.org``. The sandbox
used for M0-M3 development does not have this host in its allowlist, so
the script must be run on a machine with open internet access (or the
sandbox network policy must be updated to allow xeno-canto.org).

Usage:
    uv run --extra ml --extra dev scripts/04_xeno_canto_pull.py \\
        --species "Turdus rufiventris" --max-recordings 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bioacid.xeno_canto import (
    PseudoLabelConfig,
    SearchQuery,
    assign_pseudo_labels,
    download,
    search,
)

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="Turdus rufiventris", help="Genus species")
    parser.add_argument("--country", default="brazil")
    parser.add_argument("--min-length-s", type=int, default=5)
    parser.add_argument("--max-recordings", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "xenocanto",
        help="Root output dir; recordings go to <output>/<species_slug>/",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip audio download, just write metadata.json",
    )
    parser.add_argument("--time-window-days", type=int, default=1)
    parser.add_argument("--location-radius-m", type=float, default=500.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    species_slug = args.species.lower().replace(" ", "_")
    species_dir = args.output_dir / species_slug
    species_dir.mkdir(parents=True, exist_ok=True)

    query = SearchQuery(
        species=args.species,
        country=args.country,
        min_length_s=args.min_length_s,
    )
    print(f"querying xeno-canto: {query.as_string()}")
    t0 = time.time()
    try:
        recordings = search(query)
    except Exception as exc:
        print(f"search failed: {exc}", file=sys.stderr)
        print(
            "If this sandbox blocks xeno-canto.org, run the script on a "
            "machine with open internet access.",
            file=sys.stderr,
        )
        return 2

    recordings = recordings[: args.max_recordings]
    print(f"fetched {len(recordings)} recordings in {time.time() - t0:.1f}s")

    pseudo_labels = assign_pseudo_labels(
        recordings,
        config=PseudoLabelConfig(
            time_window_days=args.time_window_days,
            location_radius_m=args.location_radius_m,
        ),
    )

    metadata = {
        "query": query.as_string(),
        "species": args.species,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_recordings": len(recordings),
        "n_pseudo_individuals": len({lbl.individual_id for lbl in pseudo_labels.values()}),
        "recordings": [
            {
                "id": rec.id,
                "species": rec.species,
                "recordist": rec.recordist,
                "country": rec.country,
                "locality": rec.locality,
                "latitude": rec.latitude,
                "longitude": rec.longitude,
                "date": rec.date,
                "quality": rec.quality,
                "length_s": rec.length_s,
                "file_url": rec.file_url,
                "pseudo_individual_id": pseudo_labels[rec.id].individual_id,
            }
            for rec in recordings
        ],
    }
    (species_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"wrote metadata for {len(recordings)} recordings "
        f"({metadata['n_pseudo_individuals']} pseudo-individuals) "
        f"to {(species_dir / 'metadata.json').relative_to(ROOT)}"
    )

    if not args.metadata_only:
        print(f"downloading audio to {species_dir.relative_to(ROOT)}/")
        try:
            paths = download(recordings, target_dir=species_dir)
            print(f"downloaded {len(paths)} files")
        except Exception as exc:
            print(f"download failed: {exc}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

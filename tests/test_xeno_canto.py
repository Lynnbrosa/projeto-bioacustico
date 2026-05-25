"""Offline tests for bioacid.xeno_canto.

Covers the pure-Python parts: query string assembly, payload parsing, and
the pseudo-label heuristic. The network-touching ``search`` and ``download``
helpers are not exercised here.
"""

from __future__ import annotations

import pytest

from bioacid.xeno_canto import (
    PseudoLabelConfig,
    SearchQuery,
    XCRecording,
    _haversine_m,
    _within_time_window,
    assign_pseudo_labels,
)


def test_search_query_includes_quality_and_country() -> None:
    q = SearchQuery(species="Turdus rufiventris", country="brazil")
    s = q.as_string()
    assert "Turdus rufiventris" in s
    assert "cnt:brazil" in s
    assert "q:A" in s and "q:B" in s
    assert "len_gt:5" in s


def test_xcrecording_from_api_parses_length_string() -> None:
    rec = XCRecording.from_api(
        {
            "id": 12345,
            "gen": "Turdus",
            "sp": "rufiventris",
            "rec": "Lynnbrosa",
            "cnt": "Brazil",
            "loc": "Parque Ibirapuera, São Paulo",
            "lat": "-23.587",
            "lng": "-46.657",
            "date": "2024-09-15",
            "file": "https://xeno-canto.org/12345/download",
            "q": "A",
            "length": "1:23",
        }
    )
    assert rec.id == "12345"
    assert rec.species == "Turdus rufiventris"
    assert rec.length_s == 83.0
    assert rec.latitude == pytest.approx(-23.587)


def test_xcrecording_handles_malformed_length() -> None:
    rec = XCRecording.from_api({"id": 1, "gen": "T", "sp": "x", "length": "weird"})
    assert rec.length_s == 0.0


def test_haversine_zero_distance() -> None:
    assert _haversine_m(0, 0, 0, 0) == pytest.approx(0)


def test_haversine_approximate_one_degree() -> None:
    # 1 degree of latitude ≈ 111 km
    distance = _haversine_m(0, 0, 1, 0)
    assert 110_000 < distance < 112_000


def test_within_time_window_handles_iso_dates() -> None:
    assert _within_time_window("2024-09-15", "2024-09-16", window_days=1)
    assert not _within_time_window("2024-09-15", "2024-09-17", window_days=1)


def test_within_time_window_falls_back_for_bad_dates() -> None:
    assert _within_time_window("not-a-date", "not-a-date", window_days=0)
    assert not _within_time_window("not-a-date", "also-not", window_days=999)


def _make(
    rec_id: str,
    *,
    recordist: str = "alice",
    locality: str = "Park A",
    lat: float | None = -23.5,
    lng: float | None = -46.6,
    date: str = "2024-09-15",
) -> XCRecording:
    return XCRecording(
        id=rec_id,
        species="Turdus rufiventris",
        recordist=recordist,
        country="Brazil",
        locality=locality,
        latitude=lat,
        longitude=lng,
        date=date,
        file_url=f"https://xc/{rec_id}",
        quality="A",
        length_s=10.0,
    )


def test_pseudo_labels_collapse_same_recordist_same_locality() -> None:
    recordings = [_make("1"), _make("2"), _make("3")]
    labels = assign_pseudo_labels(recordings)
    ids = {labels[r.id].individual_id for r in recordings}
    assert ids == {0}, "expected a single pseudo-individual"
    assert labels["1"].member_count == 3


def test_pseudo_labels_split_different_recordists() -> None:
    recordings = [_make("1", recordist="alice"), _make("2", recordist="bob")]
    labels = assign_pseudo_labels(recordings)
    assert labels["1"].individual_id != labels["2"].individual_id


def test_pseudo_labels_split_far_locations() -> None:
    recordings = [
        _make("1", locality="Park A", lat=-23.5, lng=-46.6),
        _make("2", locality="Park B", lat=0.0, lng=0.0),  # very far away
    ]
    labels = assign_pseudo_labels(recordings)
    assert labels["1"].individual_id != labels["2"].individual_id


def test_pseudo_labels_merge_close_coordinates_diff_locality_string() -> None:
    # Same recordist, same date, different free-form locality string but
    # coordinates within 500 m (~0.001 deg lat ≈ 110 m)
    recordings = [
        _make("1", locality="entrance", lat=-23.50000, lng=-46.60000),
        _make("2", locality="lake side", lat=-23.50100, lng=-46.60050),
    ]
    labels = assign_pseudo_labels(recordings)
    assert labels["1"].individual_id == labels["2"].individual_id


def test_pseudo_labels_split_outside_time_window() -> None:
    recordings = [
        _make("1", date="2024-09-15"),
        _make("2", date="2024-09-20"),
    ]
    labels = assign_pseudo_labels(recordings, config=PseudoLabelConfig(time_window_days=1))
    assert labels["1"].individual_id != labels["2"].individual_id


def test_pseudo_labels_empty_recordist_never_merges() -> None:
    recordings = [_make("1", recordist=""), _make("2", recordist="")]
    labels = assign_pseudo_labels(recordings)
    assert labels["1"].individual_id != labels["2"].individual_id

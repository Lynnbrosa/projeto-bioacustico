"""Smoke tests: package and submodules import, version is exposed, CLI runs."""

from __future__ import annotations

import importlib

import pytest

import bioacid


def test_version_exposed() -> None:
    assert isinstance(bioacid.__version__, str)
    assert bioacid.__version__


@pytest.mark.parametrize(
    "module",
    [
        "bioacid.data",
        "bioacid.models",
        "bioacid.losses",
        "bioacid.preprocessor",
        "bioacid.train",
        "bioacid.evaluate",
        "bioacid.cli",
    ],
)
def test_submodule_importable(module: str) -> None:
    importlib.import_module(module)


def test_cli_returns_zero() -> None:
    from bioacid.cli import main

    assert main([]) == 0

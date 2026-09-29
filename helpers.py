"""Shared fixtures for the test suite."""

from gantry_ai.engine import GantryEngine


def engine():
    return GantryEngine.shared(log=lambda *a: None)

"""Every core acceptance gate (project spec 5) must report PASS."""

from __future__ import annotations

import pytest

from scr_twin_core.validation import format_report, run_all_gates


@pytest.fixture(scope="module")
def results():
    return run_all_gates(seed=0)


def test_all_gates_pass(results):
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"failed gates: {failed}\n{format_report(results)}"


def test_report_covers_all_categories(results):
    categories = {r.category for r in results}
    assert categories == {"physics", "calibration", "reliability"}


def test_report_renders(results):
    text = format_report(results)
    assert "gates passed" in text
    assert all(r.name in text for r in results)

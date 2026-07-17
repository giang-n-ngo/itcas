"""Shared abbreviated display labels for method names, reusable across reports.

Plotting code elsewhere in this package keys everything (``RunSeries.method``,
``method_styles``, discovered run buckets, ``avg_rank_row``, ...) by each
method's real, unabbreviated name -- that must never change, since it's how
runs are grouped and looked up. This module only supplies a *display* label
for legends and axis-tick text (e.g. via
``batch_vs_sequential.plot_group_grid``'s ``method_labels`` parameter), so a
caller keeps using real method names for every lookup and only swaps in the
short form at the point text is actually drawn.

Add new entries here as they're needed by future reports; :func:`abbreviate`
falls back to the method's own (real) name for anything not yet listed, so
callers never have to special-case an unmapped method.
"""
from __future__ import annotations

METHOD_ABBREVIATIONS: dict[str, str] = {
    "cas_eci": "ECI",
    "moc_cas_hard": "MOC-CAS",
    "random": "Random",
    "straddle_then_sample_lse10": "STR-TS-10",
    "bes_then_sample_lse10": "BES-TS-10",
    # "-B" (batch) distinguishes the proposed full-ITCAS/NDIG method from its
    # forced-sequential sibling itcas_seq_ndig (not yet abbreviated here --
    # add "NDIG-S" or similar if a report ever needs to show both together).
    "itcas_ndig": "NDIG-B",
}


def abbreviate(method: str) -> str:
    """Return ``method``'s short display label, or ``method`` itself if unmapped."""
    return METHOD_ABBREVIATIONS.get(method, method)

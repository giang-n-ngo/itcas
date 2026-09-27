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
    # "-B" (batch) distinguishes the proposed full-ITCAS/NDIG method
    # (QD-DPP greedy batch selection) from its forced-sequential sibling
    # itcas_seq_ndig, which gets the bare "NDIG" label -- see
    # ndig_comparison.py, which shows both together.
    "itcas_ndig": "NDIG-B",
    "itcas_seq_ndig": "NDIG",
    # Same "-B" convention extended to the other three method families' own
    # DPP-batch variants -- see method_group_comparison.py, which shows a
    # method's batch and sequential siblings in two separate reports rather
    # than side by side, but still wants the same short/consistent labels.
    "cas_eci_batch": "ECI-B",
    "moc_cas_hard_batch": "MOC-CAS-B",
    "straddle_then_sample_lse10_batch": "STR-TS-10-B",
    "bes_then_sample_lse10_batch": "BES-TS-10-B",
    # QD-DPP diversity-kernel ablations of the proposed method's own batch
    # NDIG acquisition (itcas_ndig / "NDIG-B" above) -- see
    # ndig_kernel_ablation_comparison.py, which shows all three together.
    "ndig_no_kobj_batch": "NDIG-B (no k_obj)",
    "ndig_no_kctx_batch": "NDIG-B (no k_ctx)",
    # NDIG-B's own acquisition-*component* ablations (reviewer-requested;
    # orthogonal to the QD-DPP kernel ablations above) -- see
    # ndig_b_component_ablation_comparison.py, which shows all five together.
    "itcas_ndig_no_infogain": "NDIG-B (no info gain)",
    "itcas_edig": "NDIG-B (no depth norm)",
    "itcas_efig": "NDIG-B (PoF x IG)",
    "itcas_ndig_pof_entropy": "NDIG-B (PoF entropy)",
}


def abbreviate(method: str) -> str:
    """Return ``method``'s short display label, or ``method`` itself if unmapped."""
    return METHOD_ABBREVIATIONS.get(method, method)

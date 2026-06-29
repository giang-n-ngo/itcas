"""Public algorithm API."""
from .itcas import select_batch as itcas_select_batch_discrete
from .continuous import (
    select_batch_continuous as itcas_select_batch,
    build_reference_set,
    smooth_margin,
    draw_objective_paths,
    evaluate_paths,
    multistart_ascent,
    roi_mi_quality_continuous,
)
from .roi_mi import roi_mi_quality
from .quality import (
    QUALITY_REGISTRY,
    available_qualities,
    build_quality_fn,
    efig_quality,
    edig_quality,
    ndig_quality,
    register_quality,
)
from .qd_dpp import (
    build_qd_l_ensemble,
    greedy_dpp_batch,
    rbf_kernel,
    rbf_objective_kernel,
    median_heuristic_lambda,
)

__all__ = [
    "itcas_select_batch",
    "itcas_select_batch_discrete",
    "build_reference_set",
    "smooth_margin",
    "draw_objective_paths",
    "evaluate_paths",
    "multistart_ascent",
    "roi_mi_quality_continuous",
    "roi_mi_quality",
    "efig_quality",
    "edig_quality",
    "ndig_quality",
    "build_quality_fn",
    "available_qualities",
    "register_quality",
    "QUALITY_REGISTRY",
    "build_qd_l_ensemble",
    "greedy_dpp_batch",
    "rbf_kernel",
    "rbf_objective_kernel",
    "median_heuristic_lambda",
]

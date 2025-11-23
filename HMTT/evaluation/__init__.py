"""
Evaluation modules for HMTT.

Implements the Tokenization Fidelity Score (TFS) metric.
"""

from .tfs_metric import compute_tfs, TFSMetrics

__all__ = [
    "compute_tfs",
    "TFSMetrics",
]

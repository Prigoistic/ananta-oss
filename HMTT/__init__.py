"""
Hybrid Math-Text Tokenizer (HMTT)

A discrete tokenization system that combines natural language, mathematical LaTeX,
and code tokenization with BPE vocabulary learning.

this is a purely symbolic tokenizer - NOT an encoder, VAE, or latent model.
"""

__version__ = "1.0.0"
__author__ = "Ananta Team"

from .inference.encoder import HMTTEncoder
from .inference.decoder import HMTTDecoder
from .evaluation.tfs_metric import compute_tfs

__all__ = [
    "HMTTEncoder",
    "HMTTDecoder",
    "compute_tfs",
]

"""Dataset acquisition, MATLAB reading, and EEG window preparation."""

from .corpus import (
    CorpusQualityConfig,
    audit_corpus_case,
    build_corpus_manifest,
    corpus_paths,
    load_corpus_manifest,
    write_corpus_manifest,
)
from .mat_reader import EEGCase, load_case
from .preprocess import (
    WindowedEEG,
    bis_stage,
    load_windows,
    make_windows,
    preprocess_window,
)

__all__ = [
    "EEGCase",
    "CorpusQualityConfig",
    "WindowedEEG",
    "audit_corpus_case",
    "bis_stage",
    "build_corpus_manifest",
    "corpus_paths",
    "load_case",
    "load_corpus_manifest",
    "load_windows",
    "make_windows",
    "preprocess_window",
    "write_corpus_manifest",
]

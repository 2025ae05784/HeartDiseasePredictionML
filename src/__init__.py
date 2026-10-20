"""
Source Machine Learning Operations Pipeline Package.
Provides core functions for data ingestion, clinical feature engineering,
and model experimentation.
"""

from src.data_loader import fetch_uci_dataset
from src.feature_engineering import ClinicalFeatureBuilder, process_raw_data

__all__ = [
    "fetch_uci_dataset",
    "ClinicalFeatureBuilder",
    "process_raw_data",
    "build_full_pipeline",
    "execute_experimentation_suite",
]


def __getattr__(name):
    if name in {"build_full_pipeline", "execute_experimentation_suite"}:
        from src.train import build_full_pipeline, execute_experimentation_suite

        return {
            "build_full_pipeline": build_full_pipeline,
            "execute_experimentation_suite": execute_experimentation_suite,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

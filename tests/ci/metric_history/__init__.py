"""Metric-history subsystem for the CI regression gate (port of miles@ba28991834).

One import surface for consumers, re-exporting the MetricHistoryStore contract
and its record types. Gate evaluation, selection, and constraints are imported
from their modules directly.
"""

from metric_history.storage.sqlite_store import SQLiteMetricHistoryStore
from metric_history.storage.store import MetricHistoryStore, MetricSample, RunIdentity, RunProvenance

__all__ = [
    "MetricHistoryStore",
    "MetricSample",
    "RunIdentity",
    "RunProvenance",
    "SQLiteMetricHistoryStore",
]

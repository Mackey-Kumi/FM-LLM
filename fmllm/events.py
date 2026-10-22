"""
Event-based scoring: did a warning ("the level will cross the limit within
the outlook") come true? Moved from the GridForecast app. Scores are kept per
location so a good average can't hide places where warnings fail (September
meeting, point 3).
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np


def assess_exceedance(values: np.ndarray, limit: float, margin: float = 0.0) -> Dict:
    """
    Does a trajectory (forecast or actual) reach a limit, and when?

    Args:
        values: (hours,) one sensor's values in µg/m³, hour 0 = the first forecast hour.
        limit: the level that counts as an exceedance, e.g. 35.5 µg/m³.
        margin: warn a little early: anything at or above limit - margin counts. 0 = exact limit.

    Returns:
        dict with
            exceeds: True if any hour reaches the level,
            first_hour: index of the first such hour (None if none),
            hours_above: how many hours reach it,
            peak: highest value,
            peak_hour: index of the highest value.
    """
    above = values >= limit - margin
    return {
        "exceeds": bool(above.any()),
        "first_hour": int(np.argmax(above)) if above.any() else None,
        "hours_above": int(above.sum()),
        "peak": float(np.nanmax(values)),
        "peak_hour": int(np.nanargmax(values)),
    }


def score(records: Iterable[Dict]) -> Dict:
    """
    Count how good a set of warnings was. Each record is one forecast issue
    time: did the event happen, and did we warn?

        hit = event and warned          miss = event, no warning
        false alarm = warned, no event  correct negative = neither

    Args:
        records: iterable of dicts with
            event: bool, did the level actually get reached,
            warned: bool, did the forecast say it would,
            first_hour: hours from issue to the first exceedance (used for lead time;
                only needed when event is True).

    Returns:
        dict with
            n, hits, misses, false_alarms, correct_negatives: counts,
            hit_rate: hits / (hits + misses), the share of real events we warned about,
            false_alarm_ratio: false_alarms / (hits + false_alarms), the share of our
                warnings that were wrong,
            csi: hits / (hits + misses + false_alarms), the critical success index: one
                number that punishes both misses and false alarms (1 = perfect),
            mean_lead_hours: average warning time before hit events.
        A ratio is None when its denominator is 0.
    """
    records = list(records)
    hits = sum(r["event"] and r["warned"] for r in records)
    misses = sum(r["event"] and not r["warned"] for r in records)
    false_alarms = sum(not r["event"] and r["warned"] for r in records)
    correct_neg = sum(not r["event"] and not r["warned"] for r in records)
    lead = [r["first_hour"] for r in records if r["event"] and r["warned"] and r.get("first_hour") is not None]

    def ratio(a, b) -> Optional[float]:
        """Returns: a / b, or None if b is 0."""
        return a / b if b else None

    return {
        "n": len(records),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
        "hit_rate": ratio(hits, hits + misses),
        "false_alarm_ratio": ratio(false_alarms, hits + false_alarms),
        "csi": ratio(hits, hits + misses + false_alarms),
        "mean_lead_hours": float(np.mean(lead)) if lead else None,
    }


def score_by(records: Iterable[Dict], key: str = "location") -> Dict[str, Dict]:
    """
    score(), computed separately for each group, e.g. each sensor. This keeps a
    good overall average from hiding places where the warnings fail.

    Args:
        records: as for score(), each also carrying the grouping field.
        key: name of the field to group by (default "location").

    Returns:
        {group value: score() result}, sorted by group.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return {k: score(v) for k, v in sorted(groups.items())}

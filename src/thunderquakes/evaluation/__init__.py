"""Ground-truthing model detections against open lightning catalogs (WS4)."""
from thunderquakes.evaluation.groundtruth import match_detections_to_strikes
from thunderquakes.evaluation.stats import two_proportion_ztest

__all__ = ["match_detections_to_strikes", "two_proportion_ztest"]

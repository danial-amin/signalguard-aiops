from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..metrics import TimeSeries
from ..incidents import Incident


class BaseRecipe(ABC):
    """
    High-level AIOps recipe:
    - Takes a TimeSeries (metric),
    - Runs one or more detectors,
    - Returns an Incident plus optional metadata.
    """

    @abstractmethod
    def run(self, series: TimeSeries) -> Incident:
        """
        Run the recipe on the given series and return an Incident object.
        """
        raise NotImplementedError

    def run_with_meta(self, series: TimeSeries) -> Dict[str, Any]:
        """
        Optional: return incident + extra info, e.g. detector scores, params.
        """
        incident = self.run(series)
        return {"incident": incident}

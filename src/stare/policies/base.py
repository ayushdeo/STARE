"""Policy interface. The CLOSED-LOOP distinction lives here: a policy MAY read
state.measured_expression() (expression revealed so far). One-shot competitors do NOT."""
from __future__ import annotations
import numpy as np
from ..simulator import AcquisitionState


class Policy:
    #: True if the policy conditions on already-measured expression (the closed loop).
    #: This flag is what the WS-E decisive ablation toggles.
    closed_loop: bool = False

    def select(self, state: AcquisitionState) -> int | list[int]:
        """Return the next region index (or list, for round_size>1) to profile,
        chosen from state.unprofiled."""
        raise NotImplementedError

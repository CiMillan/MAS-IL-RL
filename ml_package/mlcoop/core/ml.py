from __future__ import annotations
import numpy as np


def resolve_pending_strategies(pending_rl, pending_il, theta, rng):
    # AMBIGUITY FROM PAPER:
    # The extracted text strongly supports using theta to resolve RL-vs-IL conflict.
    # It is less clear whether the same theta also governs prior rule selection.
    # This implementation uses theta only in the final conflict-resolution step.
    same = pending_rl == pending_il
    final = pending_rl.copy()
    conflict = np.where(~same)[0]
    choose_rl = rng.random(len(conflict)) < theta
    final[conflict[~choose_rl]] = pending_il[conflict[~choose_rl]]
    return final.astype(np.int8)

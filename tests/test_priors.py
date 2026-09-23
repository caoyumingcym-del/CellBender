"""Test functions in priors.py"""

import os

import numpy as np

from cellbender.remove_background import consts
from cellbender.remove_background.data.priors import get_priors


def test_get_priors_many_cells():
    """UMI curve from a PIPseq sample with ~460k cells (MDL1856 PH20260504_1_1).
    The retry loop in get_priors used to raise low_count_threshold past every
    empty droplet and crash with an IndexError."""
    curve = np.load(os.path.join(os.path.dirname(__file__), "priors_many_cells_umi_curve.npz"))
    umi_counts = np.repeat(curve["values"], curve["counts"]).astype(np.float32)

    priors = get_priors(umi_counts=umi_counts, low_count_threshold=consts.LOW_UMI_CUTOFF)

    assert priors["cell_counts"] > priors["empty_counts"] > 0

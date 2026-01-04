import numpy as np
from research.labeler.gmm_labeler import GMMLabeler


class PipelineLabeler:
    def __init__(self, candles: np.ndarray, lag: int):
        self.labeler = GMMLabeler(candles, lag, verbose=False)

    @property
    def label_hard(self):
        return self.labeler.label_hard_state

    @property
    def label_direction(self):
        return self.labeler.label_direction_force

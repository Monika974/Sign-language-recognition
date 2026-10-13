"""Prediction smoothing for the live demo (pure Python)."""
from collections import deque


class PredictionSmoother:
    """Show a word only after the same confident prediction repeats `consensus` times.

    Stops the on-screen text flickering between classes. Pass idx=None when nothing confident was
    predicted; the current word is cleared after `consensus` consecutive empty updates.
    """

    def __init__(self, threshold: float = 0.6, consensus: int = 3):
        self.threshold = threshold
        self.history = deque(maxlen=consensus)
        self.current = None

    def update(self, idx, prob: float = 0.0):
        self.history.append(idx if (idx is not None and prob >= self.threshold) else None)
        if len(self.history) == self.history.maxlen:
            first = self.history[0]
            if all(h == first for h in self.history):
                self.current = first  # None clears the word
        return self.current

import math
import numpy as np


class LowPassFilter:
    def __init__(self):
        self.prev = None

    def apply(self, x, alpha):
        x     = np.asarray(x,    dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        if self.prev is None:
            self.prev = x.copy()
            return x.copy()
        self.prev = alpha * x + (1.0 - alpha) * self.prev
        return self.prev.copy()

    def reset(self):
        self.prev = None


class OneEuroFilter:
    """1-Euro Filter with per-element alpha (no scalar-mean collapse)."""

    def __init__(self, freq=30.0, min_cutoff=1.2, beta=0.05, d_cutoff=1.0):
        self.freq       = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)
        self.xf   = LowPassFilter()
        self.dxf  = LowPassFilter()
        self.prev = None

    def _alpha(self, cutoff):
        te  = 1.0 / max(self.freq, 1e-6)
        tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff, dtype=np.float32))
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, freq=None):
        if freq is not None and freq > 1e-6:
            self.freq = float(freq)
        x = np.asarray(x, dtype=np.float32)
        if self.prev is None:
            self.prev = x.copy()
            self.xf.apply(x, self._alpha(self.min_cutoff))
            return x.copy()
        dx        = (x - self.prev) * self.freq
        self.prev = x.copy()
        edx       = self.dxf.apply(dx, self._alpha(self.d_cutoff))
        cutoff    = self.min_cutoff + self.beta * np.abs(edx)
        return self.xf.apply(x, self._alpha(cutoff))

    def reset(self):
        self.xf.reset()
        self.dxf.reset()
        self.prev = None

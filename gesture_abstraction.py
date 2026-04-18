from collections import deque
import numpy as np

# InterHand tip→wrist layout inside each 21-kp block
WRIST = 20
FINGER_CHAINS_IDX = [
    [20,  3,  2,  1,  0],   # thumb
    [20,  7,  6,  5,  4],   # index
    [20, 11, 10,  9,  8],   # middle
    [20, 15, 14, 13, 12],   # ring
    [20, 19, 18, 17, 16],   # pinky
]
FINGER_TIPS = [0,  4,  8, 12, 16]
FINGER_MCPS = [3,  7, 11, 15, 19]


class GestureAbstractor:
    """
    OPEN PALM vs GRASP — strongly biased toward GRASP.

    Metric: tip/MCP distance ratio = dist(tip,wrist) / dist(MCP,wrist)
      Fully flat open palm : ~2.8 – 4.0
      Slight curl (pen)    : ~1.6 – 2.4  → GRASP
      Full fist            : ~0.8 – 1.5  → GRASP

    Rules
    -----
    • Default state: GRASP (bias).
    • Flip to OPEN PALM only when ALL of:
        - mean ratio across all non-thumb fingers > OPEN_MEAN_THR  (2.8)
        - every individual non-thumb finger ratio > OPEN_EACH_THR  (2.4)
        - at least MIN_OPEN_FRAMES consecutive frames agree
    • Return to GRASP the moment any single finger drops below CURL_THR (2.6).
    • History shortened to 5 frames for fast response.

    This means a pen-hold, knife-grip, partial curl, or any ambiguous
    pose always reads as GRASP. Only a clearly flat, fully-extended open
    palm registers as OPEN PALM.
    """

    CURL_THR       = 2.6   # any finger below this → GRASP immediately
    OPEN_MEAN_THR  = 2.8   # mean ratio needed to even consider OPEN PALM
    OPEN_EACH_THR  = 2.4   # every non-thumb finger must exceed this
    MIN_OPEN_FRAMES = 4    # consecutive frames of "open" needed to flip

    def __init__(self, history=5):
        self.history      = deque(maxlen=history)
        self._current     = "GRASP"   # start as GRASP
        self._open_streak = 0

    def classify(self, keypoints, scores, score_thr=0.12):
        kp = np.asarray(keypoints, dtype=np.float32)
        sc = np.asarray(scores,    dtype=np.float32)

        if kp.shape[0] < 21 or np.mean(sc) < score_thr:
            self.history.clear()
            self._current     = "GRASP"
            self._open_streak = 0
            return "GRASP", {}

        wrist  = kp[WRIST, :2]
        ratios = []   # all non-thumb fingers with good confidence
        for fi, (tip_i, mcp_i) in enumerate(zip(FINGER_TIPS, FINGER_MCPS)):
            if fi == 0:           # skip thumb — geometry differs
                continue
            if sc[tip_i] < score_thr or sc[mcp_i] < score_thr:
                continue
            d_tip = np.linalg.norm(kp[tip_i, :2] - wrist)
            d_mcp = np.linalg.norm(kp[mcp_i, :2] - wrist)
            if d_mcp < 1e-3:
                continue
            ratios.append(d_tip / d_mcp)

        if not ratios:
            # Can't compute — stay GRASP
            self._open_streak = 0
            return "GRASP", {"mean_ratio": 0, "n_curled": 0, "confidence": 1.0}

        mean_ratio = float(np.mean(ratios))
        n_curled   = int(sum(r < self.CURL_THR for r in ratios))
        all_open   = all(r > self.OPEN_EACH_THR for r in ratios)

        # Any curl → immediate GRASP, reset streak
        if n_curled > 0:
            self._open_streak = 0
            raw = "GRASP"
        elif mean_ratio > self.OPEN_MEAN_THR and all_open:
            self._open_streak += 1
            raw = "OPEN PALM" if self._open_streak >= self.MIN_OPEN_FRAMES else "GRASP"
        else:
            self._open_streak = 0
            raw = "GRASP"

        # Hysteresis: once OPEN PALM, require a curl to leave
        if self._current == "OPEN PALM" and n_curled == 0:
            raw = "OPEN PALM"

        self.history.append(raw)
        vals  = list(self.history)
        label = max(set(vals), key=vals.count)
        conf  = vals.count(label) / len(vals)
        self._current = label

        return label, {
            "mean_ratio": mean_ratio,
            "n_curled":   n_curled,
            "confidence": conf,
        }

from pathlib import Path
import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown

# InterHand2.6M 42-kp layout (tip→wrist within each 21-kp block)
# Right hand: 0-20    Left hand: 21-41
# 0/21=thumb_tip  3/24=thumb_MCP  4/25=index_tip  7/28=index_MCP
# 8/29=mid_tip   11/32=mid_MCP  12/33=ring_tip  15/36=ring_MCP
# 16/37=pinky_tip 19/40=pinky_MCP  20/41=wrist


class MMPoseHandBackend:
    """
    Dual-hand tracker using InterNet (InterHand2.6M, ECCV 2020).

    Depth-aided tracking (when calibrated)
    ---------------------------------------
    After depth calibration the tracker uses the depth at each detected
    hand's palm centre to improve slot matching. Two hands at the same
    2D position but different depths (e.g. one hand behind the other)
    are disambiguated using depth. The slot score function penalises
    detections whose depth differs significantly from the slot's last
    known depth — this prevents the tracker swapping slots when hands
    cross in 2D.

    Slot 0 = right hand, slot 1 = left hand.
    EMA on centre/size/depth — 12-frame miss tolerance.
    """

    DEPTH_W = 0.4   # weight of depth consistency in slot scoring (0=ignore)

    def __init__(self, device="cpu", score_thr=0.12, infer_scale=0.6):
        repo   = Path(__file__).resolve().parents[3]
        config = str(repo / "configs/hand_3d_keypoint/internet/interhand3d"
                          "/internet_res50_4xb16-20e_interhand3d-256x256.py")
        ckpt   = str(repo / "checkpoints/res50.pth")
        self.model       = init_model(config, ckpt, device=device)
        self.score_thr   = score_thr
        self.infer_scale = infer_scale

        self._centers = [None, None]
        self._sizes   = [None, None]
        self._depths  = [None, None]   # EMA palm depth per slot (metres)
        self._missing = [0, 0]
        self.MAX_MISS = 12

    # ── public ────────────────────────────────────────────────────────────
    def infer(self, frame_bgr, depth_info=None):
        """
        frame_bgr  : camera frame
        depth_info : dict with keys 'depth_map'(H,W float32) and
                     'depth_est' (DepthEstimator instance), or None.
                     When provided, palm depth is used in slot scoring.
        Returns list of 0-2 hand dicts.
        """
        raw = self._run(frame_bgr)
        raw = _dedup(raw)

        # Attach depth to each raw detection if available
        if depth_info is not None:
            dm  = depth_info["depth_map"]
            est = depth_info["depth_est"]
            for h in raw:
                h["palm_depth"] = est.depth_at_hand(
                    dm, h["keypoints"][:, :2], h["scores"])
        else:
            for h in raw:
                h["palm_depth"] = None

        out = self._match(raw)

        for slot, hand in enumerate(out):
            if hand is not None:
                self._missing[slot] = 0
                self._ema(slot, hand)
            else:
                self._missing[slot] += 1
                if self._missing[slot] >= self.MAX_MISS:
                    self._centers[slot] = self._sizes[slot] = None
                    self._depths[slot]  = None

        return [h for h in out if h is not None]

    # ── inference ─────────────────────────────────────────────────────────
    def _run(self, frame_bgr):
        scaled  = cv2.resize(frame_bgr, None,
                             fx=self.infer_scale, fy=self.infer_scale)
        results = inference_topdown(self.model, scaled)
        return self._extract(results)

    def _extract(self, results):
        hands  = []
        sc_min = self.score_thr * 0.5
        for res in results:
            pred = getattr(res, "pred_instances", None)
            if pred is None:
                continue
            kpa = np.asarray(getattr(pred, "keypoints",      []))
            sca = np.asarray(getattr(pred, "keypoint_scores", []))
            if kpa.ndim < 3 or kpa.shape[0] == 0:
                continue
            kp = kpa[0].astype(np.float32)
            sc = sca[0].astype(np.float32)
            if sc.max() > 1.0:
                sc /= 255.0
            kp[:, 0] /= self.infer_scale
            kp[:, 1] /= self.infer_scale
            n = kp.shape[0]
            if n == 42:
                for start, side in [(0, "right"), (21, "left")]:
                    hkp = kp[start:start+21].copy()
                    hsc = sc[start:start+21].copy()
                    if np.mean(hsc) >= sc_min:
                        hands.append(_mkhand(hkp, hsc, side))
            elif n == 21 and np.mean(sc) >= sc_min:
                hands.append(_mkhand(kp, sc, "unknown"))
        return hands

    # ── slot matching ─────────────────────────────────────────────────────
    def _match(self, raw):
        out, used = [None, None], set()
        # Pass 1: side label
        for j, h in enumerate(raw):
            slot = {"right": 0, "left": 1}.get(h["hand_side"])
            if slot is not None and out[slot] is None:
                out[slot] = h
                used.add(j)
        # Pass 2: proximity + depth consistency
        for slot in range(2):
            if out[slot] is not None:
                continue
            best_s, best_j = -1e9, None
            for j, h in enumerate(raw):
                if j in used:
                    continue
                s = self._score(h, slot)
                if s > best_s:
                    best_s, best_j = s, j
            if best_j is not None and raw[best_j]["mean_score"] >= self.score_thr:
                out[slot] = raw[best_j]
                used.add(best_j)
        return out

    def _score(self, hand, slot):
        conf = hand["mean_score"]
        if self._centers[slot] is None:
            return conf

        # 2D proximity penalty
        sz    = self._sizes[slot] or 150.0
        dist  = np.linalg.norm(hand["keypoints"][:, :2].mean(0) - self._centers[slot])
        score = conf - 0.5 * (dist / sz)

        # Depth consistency bonus/penalty
        if (self._depths[slot] is not None and
                hand.get("palm_depth") is not None):
            depth_diff = abs(hand["palm_depth"] - self._depths[slot])
            # Penalise if depth differs by more than 15 cm
            score -= self.DEPTH_W * max(0.0, depth_diff - 0.15)

        return score

    # ── EMA state update ──────────────────────────────────────────────────
    def _ema(self, slot, hand, alpha=0.35):
        kp = hand["keypoints"][:, :2]
        nc = kp.mean(0)
        d  = kp.max(0) - kp.min(0)
        ns = float(max(d[0], d[1], 100.0))
        if self._centers[slot] is None:
            self._centers[slot] = nc
            self._sizes[slot]   = ns
        else:
            self._centers[slot] = alpha*nc + (1-alpha)*self._centers[slot]
            self._sizes[slot]   = alpha*ns + (1-alpha)*self._sizes[slot]

        # Update depth EMA
        pd = hand.get("palm_depth")
        if pd is not None:
            if self._depths[slot] is None:
                self._depths[slot] = pd
            else:
                self._depths[slot] = alpha*pd + (1-alpha)*self._depths[slot]


# ── helpers ───────────────────────────────────────────────────────────────

def _mkhand(kp, sc, side):
    if kp.shape[1] == 2:
        kp = np.concatenate([kp, np.zeros((21,1), np.float32)], axis=1)
    return {"keypoints": kp[:,:3].copy(), "scores": sc.copy(),
            "mean_score": float(np.mean(sc)), "hand_side": side,
            "palm_depth": None}


def _dedup(hands, thr=0.5):
    if len(hands) <= 1:
        return hands
    boxes = [(*h["keypoints"][:,:2].min(0), *h["keypoints"][:,:2].max(0))
             for h in hands]
    keep, sup = [], set()
    for i in range(len(hands)):
        if i in sup:
            continue
        keep.append(i)
        for j in range(i+1, len(hands)):
            if j in sup:
                continue
            if _iou(boxes[i], boxes[j]) > thr:
                if hands[j]["mean_score"] > hands[keep[-1]]["mean_score"]:
                    keep[-1] = j
                sup.add(j)
    return [hands[k] for k in keep]


def _iou(a, b):
    ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
    ix = max(0.0, min(ax1,bx1)-max(ax0,bx0))
    iy = max(0.0, min(ay1,by1)-max(ay0,by0))
    inter = ix*iy
    if inter == 0: return 0.0
    return inter/((ax1-ax0)*(ay1-ay0)+(bx1-bx0)*(by1-by0)-inter+1e-9)

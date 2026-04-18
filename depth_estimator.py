from pathlib import Path
import cv2
import numpy as np
import torch
from .monodepth2_networks import DepthDecoder, ResnetEncoder

TARGET_M = 0.40   # palm held at 40 cm during calibration


class DepthEstimator:
    """
    Monodepth2 monocular depth with one-shot palm calibration.

    Usage
    -----
    1. Call estimate(frame) every frame → (smoothed_depth, raw_depth)
    2. When user presses 'c' with open palm at 40 cm, call:
           calibrate(raw_depth, palm_kp2d)
       This computes a scale factor so depth values become real metres.
    3. Use sample(depth_map, xy) to read depth at any pixel.
    4. Use depth_at_hand(depth_map, kp2d, sc) to get palm-centre depth
       for a detected hand — used by the tracker for depth-aided matching.

    If model weights are missing, falls back to a luminance-based proxy
    so the rest of the pipeline still runs.
    """

    def __init__(self, model_path, device="cpu", min_depth=0.1, max_depth=2.5):
        self.device    = device
        self.path      = Path(model_path)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.feed_h    = 192
        self.feed_w    = 640
        self.encoder   = None
        self.decoder   = None
        self.enabled   = False
        self._prev     = None
        self.calib_scale = None   # None = uncalibrated
        self._load()

    def _load(self):
        ep = self.path / "encoder.pth"
        dp = self.path / "depth.pth"
        if not ep.exists() or not dp.exists():
            return
        self.encoder = ResnetEncoder(18, False)
        ed = torch.load(str(ep), map_location=self.device)
        self.feed_h = int(ed.get("height", 192))
        self.feed_w = int(ed.get("width",  640))
        self.encoder.load_state_dict(
            {k: v for k, v in ed.items() if k in self.encoder.state_dict()},
            strict=False)
        self.encoder.to(self.device).eval()
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))
        self.decoder.load_state_dict(
            torch.load(str(dp), map_location=self.device), strict=False)
        self.decoder.to(self.device).eval()
        self.enabled = True

    # ── core estimate ──────────────────────────────────────────────────────
    def estimate(self, frame_bgr):
        """Returns (smoothed_depth, raw_depth), both (H,W) float32."""
        if not self.enabled:
            d = self._fallback(frame_bgr)
            return d, d

        H, W = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp  = cv2.resize(rgb, (self.feed_w, self.feed_h)).astype(np.float32) / 255.0
        x    = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            disp = self.decoder(self.encoder(x))[("disp",0)].squeeze().cpu().numpy()

        disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR)
        disp = cv2.GaussianBlur(disp, (7, 7), 0)
        raw  = (1.0 / np.clip(disp, 1e-6, None)).astype(np.float32)

        if self.calib_scale is not None:
            metres = np.clip(raw * self.calib_scale, self.min_depth, self.max_depth)
        else:
            # Normalise to display range so inset looks sensible even uncalibrated
            n      = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-6)
            metres = (self.min_depth + n * (self.max_depth - self.min_depth)).astype(np.float32)

        # Temporal EMA smoothing
        if self._prev is None:
            self._prev = metres.copy()
        else:
            self._prev = (0.8 * self._prev + 0.2 * metres).astype(np.float32)

        return self._prev.copy(), raw

    # ── calibration ────────────────────────────────────────────────────────
    def calibrate(self, raw_depth, palm_kp2d):
        """
        Compute scale so palm-centre raw depth == 0.40 m.
        palm_kp2d : (21,2) open-palm keypoints (in original frame pixels).
        Returns True on success.
        """
        H, W = raw_depth.shape
        cx   = int(np.clip(palm_kp2d[:, 0].mean(), 0, W - 1))
        cy   = int(np.clip(palm_kp2d[:, 1].mean(), 0, H - 1))
        r    = 25
        patch = raw_depth[max(0,cy-r):min(H,cy+r), max(0,cx-r):min(W,cx+r)]
        med   = float(np.median(patch))
        if med > 1e-6:
            self.calib_scale = TARGET_M / med
            print(f"[Depth] calibrated  raw_med={med:.4f}  "
                  f"scale={self.calib_scale:.4f}  → 1 unit = {self.calib_scale:.3f} m")
            return True
        print("[Depth] calibration failed — no depth at palm centre")
        return False

    # ── helpers ────────────────────────────────────────────────────────────
    def sample(self, depth_map, xy, patch=10):
        """Median depth in a patch around pixel (x,y)."""
        H, W = depth_map.shape
        x = int(np.clip(round(float(xy[0])), 0, W-1))
        y = int(np.clip(round(float(xy[1])), 0, H-1))
        return float(np.median(
            depth_map[max(0,y-patch):min(H,y+patch+1),
                      max(0,x-patch):min(W,x+patch+1)]))

    def depth_at_hand(self, depth_map, kp2d, sc, joint_thr=0.12):
        """
        Returns median depth at palm centre for a detected hand.
        Used by the tracker to include depth in slot scoring.
        Returns None if no confident palm joints visible.
        """
        palm_ids = [20, 7, 11, 15, 19]
        valid    = [j for j in palm_ids if sc[j] > joint_thr]
        if not valid:
            return None
        cx = float(np.mean(kp2d[valid, 0]))
        cy = float(np.mean(kp2d[valid, 1]))
        return self.sample(depth_map, (cx, cy), patch=12)

    @staticmethod
    def colorize(depth):
        d = depth - depth.min()
        d = (d / max(depth.max() - depth.min(), 1e-6) * 255).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)

    def _fallback(self, frame_bgr):
        g  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        h, w = g.shape
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        c  = 0.6*(1-g) + 0.4*yy
        c  = (c-c.min()) / max(c.max()-c.min(), 1e-6)
        return (self.min_depth + c*(self.max_depth-self.min_depth)).astype(np.float32)

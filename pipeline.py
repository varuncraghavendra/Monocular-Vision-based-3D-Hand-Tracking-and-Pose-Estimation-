import os, time
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from .camera              import ThreadedCamera
from .depth_estimator     import DepthEstimator
from .gesture_abstraction import GestureAbstractor, FINGER_CHAINS_IDX, WRIST
from .one_euro_filter     import OneEuroFilter
from .pose_backends       import MMPoseHandBackend

# ── Colours — identical on camera overlay AND 3D plot ────────────────────
#           thumb           index           middle          ring            pinky
_BGR = [(203,192,255), (50,100,255), (50,205,50), (0,165,255), (147,112,219)]
_RGB = [(r/255, g/255, b/255) for (b, g, r) in _BGR]
_PALM_BGR = (220, 220, 220)

# ── Visibility thresholds ────────────────────────────────────────────────
_MEAN_THR  = 0.18
_JOINT_THR = 0.12
_FRAC_THR  = 0.55

# ── Layout ───────────────────────────────────────────────────────────────
CAM_W, CAM_H   = 960, 720
PLOT_W, PLOT_H = 640, 720
CALIB_SECS     = 3          # countdown before depth sample is taken


class RobotLearningHandPipeline:
    """
    Left panel  : 960×720 camera + per-finger thin coloured skeleton.
    Right panel : 640×720 real-time 3D hand pose graph (matplotlib).

    Depth calibration
    -----------------
    Press 'c' → hold open palm at exactly 40 cm from camera.
    A 3-second countdown then samples the depth map at the palm centre
    and computes a scale factor → all subsequent depth readings are in
    real metres.  After calibration the tracker also uses depth to
    improve slot matching when hands cross in 2D.
    Press 'r' to reset calibration.

    Keys: ESC=quit  c=calibrate  r=reset
    """

    def __init__(self, device="cpu", score_thr=0.12,
                 depth_model=None, infer_scale=0.6):
        self.cam   = ThreadedCamera(width=CAM_W, height=CAM_H)
        self.depth = DepthEstimator(depth_model or "", device=device)
        self.pose  = MMPoseHandBackend(device=device,
                                       score_thr=score_thr,
                                       infer_scale=infer_scale)
        self.score_thr = score_thr

        self.filters  = [
            OneEuroFilter(freq=25, min_cutoff=1.2, beta=0.05),
            OneEuroFilter(freq=25, min_cutoff=1.2, beta=0.05),
        ]
        self.gestures = [
            GestureAbstractor(history=5),
            GestureAbstractor(history=5),
        ]

        self._fig = plt.figure(figsize=(PLOT_W/100, PLOT_H/100),
                               dpi=100, facecolor="white")
        self._ax  = self._fig.add_subplot(111, projection="3d")

        self.last_t       = time.time()
        self.fps          = 0.0
        self.calib_active = False
        self.calib_start  = None
        self.calib_done   = False

    # ── FPS ───────────────────────────────────────────────────────────────
    def _tick(self):
        now = time.time(); dt = now - self.last_t; self.last_t = now
        if dt > 1e-6:
            self.fps = (1/dt) if self.fps == 0 else (0.8*self.fps + 0.2/dt)
        return self.fps

    def _visible(self, sc):
        return (float(np.mean(sc)) >= _MEAN_THR and
                float(np.mean(sc > _JOINT_THR)) >= _FRAC_THR)

    # ── Camera overlay ────────────────────────────────────────────────────
    def _draw_skeleton(self, frame, kp2d, sc):
        for fi, chain in enumerate(FINGER_CHAINS_IDX):
            col = _BGR[fi]
            pts = [(int(round(kp2d[j,0])), int(round(kp2d[j,1])))
                   for j in chain if sc[j] > _JOINT_THR]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(frame, a, b, col, 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, col, -1, cv2.LINE_AA)
        for a, b in zip([20,7,11,15], [7,11,15,19]):
            if sc[a] > _JOINT_THR and sc[b] > _JOINT_THR:
                cv2.line(frame,
                         tuple(np.round(kp2d[a]).astype(int)),
                         tuple(np.round(kp2d[b]).astype(int)),
                         _PALM_BGR, 1, cv2.LINE_AA)

    def _draw_label(self, frame, slot, kp2d, sc, label, meta):
        if sc[WRIST] > _JOINT_THR:
            ax_, ay_ = kp2d[WRIST]
        else:
            vis = kp2d[sc > _JOINT_THR]
            if not len(vis): return
            ax_, ay_ = vis.mean(0)
        H, W  = frame.shape[:2]
        col   = _BGR[slot % len(_BGR)]
        conf  = int(meta.get("confidence", 0) * 100)
        n_c   = meta.get("n_curled", 0)
        txt   = f"{'R' if slot==0 else 'L'}: {label} {conf}% curl:{n_c}/4"
        fx    = int(np.clip(ax_-70, 2, W-230))
        fy    = int(np.clip(ay_+22, 2, H-24))
        ov    = frame.copy()
        cv2.rectangle(ov, (fx,fy), (fx+222,fy+20), (10,10,10), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, txt, (fx+4,fy+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    def _draw_depth_label(self, frame, slot, kp2d, sc, depth_map):
        """Show palm-centre depth next to each hand."""
        palm = [20, 7, 11, 15, 19]
        v    = [j for j in palm if sc[j] > _JOINT_THR]
        if not v: return
        cx = float(np.mean(kp2d[v, 0]))
        cy = float(np.mean(kp2d[v, 1]))
        d  = self.depth.sample(depth_map, (cx, cy), patch=10)
        suf = "m" if self.calib_done else "m*"
        col = _BGR[slot % len(_BGR)]
        H, W = frame.shape[:2]
        tx = int(np.clip(cx-55, 2, W-150))
        ty = int(np.clip(cy-22, 16, H-2))
        cv2.putText(frame, f"{'R' if slot==0 else 'L'} {d:.2f}{suf}",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)

    def _draw_depth_inset(self, frame, depth):
        col   = self.depth.colorize(depth)
        inset = cv2.resize(col, (140, 105))
        h     = frame.shape[0]
        frame[h-115:h-10, 10:150] = inset
        cv2.rectangle(frame, (10,h-115), (150,h-10), (140,140,140), 1)
        label = "Depth CAL" if self.calib_done else "Depth UNCAL"
        cv2.putText(frame, label, (14,h-101),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (80,220,80) if self.calib_done else (180,180,180), 1)

    def _draw_calib_ui(self, frame, hands):
        elapsed   = time.time() - self.calib_start
        remaining = max(0.0, CALIB_SECS - elapsed)
        H, W      = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (W,H), (0,0,0), -1)
        cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)
        cv2.putText(frame, "CALIBRATION  hold open palm at 40 cm",
                    (W//2-300, H//2-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,220,60), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Capturing in {remaining:.1f}s ...",
                    (W//2-130, H//2+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)
        bw = int((elapsed / CALIB_SECS) * (W-80))
        cv2.rectangle(frame, (40,H//2+42), (W-40,H//2+58), (50,50,50), -1)
        cv2.rectangle(frame, (40,H//2+42), (40+bw,H//2+58), (60,200,60), -1)
        # Return palm keypoints when countdown finishes AND a hand is visible
        if elapsed >= CALIB_SECS and hands:
            return hands[0]["keypoints"][:, :2]
        return None

    def _draw_hud(self, frame, n):
        H, W = frame.shape[:2]
        cal  = "CAL" if self.calib_done else "UNCAL"
        cv2.putText(frame, f"FPS:{self.fps:.1f}  Hands:{n}  Depth:{cal}",
                    (12, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160,160,160), 1)
        cv2.putText(frame, "c=calibrate  r=reset  ESC=quit",
                    (W-260, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120,120,120), 1)

    # ── 3D pose plot ──────────────────────────────────────────────────────
    def _render_3d(self, hands_data, title):
        ax = self._ax; ax.cla()
        ax.set_facecolor("white")
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("#cccccc")
        ax.grid(True, color="#e0e0e0", linewidth=0.3)
        ax.tick_params(labelsize=5)
        ax.set_title(title, fontsize=9, pad=2)

        all_pts = []
        for kp, sc in hands_data:
            has_z = kp.shape[1] >= 3 and np.any(kp[:, 2] != 0)
            px = kp[:, 0].copy()
            py = kp[:, 2].copy() if has_z else np.zeros(21, np.float32)
            pz = -kp[:, 1].copy()
            all_pts.append(np.stack([px, py, pz], axis=1))

            for fi, chain in enumerate(FINGER_CHAINS_IDX):
                col  = _RGB[fi]
                good = [j for j in chain if sc[j] > _JOINT_THR]
                if len(good) < 2: continue
                ax.plot([px[j] for j in good], [py[j] for j in good],
                        [pz[j] for j in good],
                        color=col, linewidth=1.8, solid_capstyle="round")
                ax.scatter([px[j] for j in good], [py[j] for j in good],
                           [pz[j] for j in good],
                           color=col, s=12, zorder=5, depthshade=False)

            pb = [j for j in [20,7,11,15,19] if sc[j] > _JOINT_THR]
            if len(pb) >= 2:
                ax.plot([px[j] for j in pb], [py[j] for j in pb],
                        [pz[j] for j in pb],
                        color=(0.75,0.75,0.75), linewidth=1.0)

        if all_pts:
            pts = np.concatenate(all_pts, axis=0)
            for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                lo, hi = pts[:,dim].min(), pts[:,dim].max()
                pad    = max((hi-lo)*0.28, 40.0)
                setter(lo-pad, hi+pad)

        ax.set_xlabel("X",       fontsize=7, labelpad=1)
        ax.set_ylabel("Z depth", fontsize=7, labelpad=1)
        ax.set_zlabel("Y",       fontsize=7, labelpad=1)

        self._fig.canvas.draw()
        buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
        return cv2.cvtColor(cv2.resize(buf, (PLOT_W, PLOT_H)), cv2.COLOR_RGB2BGR)

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        blank = self._render_3d([], "Prediction (0)")

        while True:
            fps_now = self._tick()

            frame = self.cam.read()
            if frame is None:
                continue
            if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H))

            vis = frame.copy()

            # ── Depth estimate (every frame, before hand detection)
            depth_map, raw_depth = self.depth.estimate(frame)
            self._draw_depth_inset(vis, depth_map)

            # ── Hand detection — pass depth info when calibrated
            depth_info = None
            if self.calib_done:
                depth_info = {"depth_map": depth_map, "depth_est": self.depth}
            hands = self.pose.infer(frame, depth_info=depth_info)

            # ── Calibration countdown UI
            if self.calib_active:
                ckp = self._draw_calib_ui(vis, hands)
                if ckp is not None:
                    if self.depth.calibrate(raw_depth, ckp):
                        self.calib_done = True
                    self.calib_active = False

            # ── Per-hand processing
            hands_3d = []
            if hands:
                for slot, hand in enumerate(hands):
                    kp = hand["keypoints"].copy()   # (21, 3)
                    sc = hand["scores"].copy()

                    # Smooth XY; z from InterNet model
                    kp[:, :2] = self.filters[slot](kp[:, :2],
                                                   freq=max(fps_now, 10.0))

                    if self._visible(sc):
                        self._draw_skeleton(vis, kp[:, :2], sc)
                        self._draw_depth_label(vis, slot, kp[:, :2], sc, depth_map)

                    label, meta = self.gestures[slot].classify(
                        kp, sc, score_thr=self.score_thr)

                    if self._visible(sc):
                        self._draw_label(vis, slot, kp[:, :2], sc, label, meta)

                    hands_3d.append((kp, sc))

                for slot in range(2):
                    if slot >= len(hands):
                        self.filters[slot].reset()
                        self.gestures[slot].history.clear()
                        self.gestures[slot]._current = "GRASP"
                        self.gestures[slot]._open_streak = 0
            else:
                for f in self.filters:
                    f.reset()
                for g in self.gestures:
                    g.history.clear()
                    g._current     = "GRASP"
                    g._open_streak = 0
                if not self.calib_active:
                    cv2.putText(vis, "NO HAND DETECTED", (12, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140,140,140), 2)

            self._draw_hud(vis, len(hands))

            n    = len(hands_3d)
            plot = self._render_3d(hands_3d, f"Prediction ({n})") if n else blank

            canvas = np.ones((CAM_H, CAM_W+PLOT_W, 3), dtype=np.uint8) * 235
            canvas[:, :CAM_W]                                  = vis
            canvas[:plot.shape[0], CAM_W:CAM_W+plot.shape[1]] = plot

            cv2.imshow("PRCV Hand Tracking", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('c') and not self.calib_active:
                self.calib_active = True
                self.calib_start  = time.time()
            elif key == ord('r'):
                self.depth.calib_scale = None
                self.calib_done        = False
                self.calib_active      = False
                # Reset tracker depth memory
                self.pose._depths = [None, None]
                print("[Depth] calibration reset.")

        plt.close(self._fig)
        self.cam.release()
        cv2.destroyAllWindows()

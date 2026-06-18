#!/usr/bin/env python3
"""Temporal exo-fallback — recover ego-blind frames using the exo cameras.

When the ego camera loses a hand, the ego anchor is unavailable, but ≥2 exo
cameras often still see it. We propagate the hand's 3D identity from the last good
frame: predict its position, reproject into each exo view, match the nearest
detection (bystanders are far from the prediction → rejected), and triangulate
exo-only. The new 3D becomes the next frame's prediction (frame-by-frame tracking
through the gap). A reprojection-consistency check stops the track if it diverges.

Outputs (separate from the ego-only run, for comparison):
  j3d_trajectory_tracked.npz, figure7_video_tracked.mp4, figure7_grid_tracked.jpg
Run with:  ../.venv-egoexo/bin/python stage3_tracked.py
"""
import json, subprocess
from pathlib import Path
import numpy as np
import cv2

from triangulate_geom import load_gopro_cams, draw, CALIB, FRAMES
from reproject_ego import build_ego_camera, TRAJ
from stage3_auto import ExoCam, EgoCam, match_exo, robust_triangulate
from stage3_video import Traj, dets_for, KPTS_ALL, NFRAMES, WORK, PANEL

MATCH_FRAC = 0.06     # exo match radius vs prediction (fraction of image dim)
MAX_TRACK = 60        # stop tracking after this many consecutive ego-blind frames


def match_exo_pred(pred_J3D, exo_cams, exo_dets):
    """Match each exo detection to the predicted 3D hand by reprojection proximity."""
    accepted = []
    for name, cam in exo_cams.items():
        pred2d = np.array([cam.project(pred_J3D[j]) for j in range(21)])
        best, bestd = None, np.inf
        for d in exo_dets.get(name, []):
            dist = float(np.mean(np.linalg.norm(d - pred2d, axis=1)))
            if dist < bestd:
                bestd, best = dist, d
        if best is not None and bestd < MATCH_FRAC * max(cam.c["WH"]):
            accepted.append((cam, best))
    return accepted


def main():
    exo = {v: ExoCam(c, v) for v, c in load_gopro_cams(CALIB).items()}
    kp = np.load(KPTS_ALL)
    oc = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")]
    tr = Traj(TRAJ / "closed_loop_trajectory.csv")
    PANEL.mkdir(parents=True, exist_ok=True)

    last = {"R": None, "L": None}; last_f = {"R": -999, "L": -999}
    stats = {"ego": 0, "fallback": 0, "skip": 0}
    augmented, with_hand = {}, []
    for f in range(NFRAMES):
        cam, R_dc, t_dc = build_ego_camera(oc[f])
        R_wd, t_wd = tr.pose(oc[f]["tracking_timestamp_us"])
        egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd)
        ego_dets = dets_for(kp, "aria01", f)
        exo_dets = {v: [k for k, m in dets_for(kp, v, f)] for v in exo}
        img = cv2.imread(str(FRAMES / "aria01" / f"frame_{f:06d}.jpg"))
        drew = False

        for hand, want in (("R", 1), ("L", 0)):
            J3D, mode = None, None
            # 1) ego-anchored (preferred)
            ego_h = next((k for k, m in ego_dets if int(m[0]) == want), None)
            if ego_h is not None:
                acc, _ = match_exo(egocam.unproject, ego_h, exo, exo_dets)
                views = [(egocam, ego_h)] + acc
                if len(views) >= 2:
                    try:
                        J, _, med = robust_triangulate(views, ego_idx=0)
                        if med.get(0, 1e9) <= 40:
                            J3D, mode = J, "ego"
                    except np.linalg.LinAlgError:
                        pass
            # 2) temporal exo-fallback (ego blind but exo sees it)
            if J3D is None and last[hand] is not None and (f - last_f[hand]) <= MAX_TRACK:
                acc = match_exo_pred(last[hand], exo, exo_dets)
                if len(acc) >= 2:
                    try:
                        J, _, med = robust_triangulate(acc, ego_idx=None)
                        if np.median(list(med.values())) <= max(c.tau for c, _ in acc):
                            J3D, mode = J, "fallback"
                    except np.linalg.LinAlgError:
                        pass

            if J3D is not None:
                proj = np.array([egocam.project(J3D[j]) for j in range(21)])
                if np.all(np.isfinite(proj)) and np.abs(proj).max() < 1e5:
                    draw(img, proj)
                augmented[f"{f:06d}_{hand}"] = J3D
                last[hand] = J3D; last_f[hand] = f
                stats[mode] += 1; drew = True
        if drew:
            with_hand.append(f)
        else:
            stats["skip"] += 1
        cv2.imwrite(str(PANEL / f"frame_{f:06d}.jpg"), img)
        if f % 100 == 0:
            print(f"  frame {f:4d}: ego={stats['ego']} fallback={stats['fallback']} skip={stats['skip']}")

    np.savez_compressed(WORK / "j3d_trajectory_tracked.npz", **augmented)
    nR = sum(1 for k in augmented if k.endswith("_R"))
    print(f"\ndone. right hands: {nR}/{NFRAMES} ({100*nR/NFRAMES:.0f}%)  "
          f"[ego-anchored + exo-fallback]")
    print(f"frames with ≥1 hand: {len(with_hand)}/{NFRAMES} ({100*len(with_hand)/NFRAMES:.0f}%)")
    print(f"  ego-anchored hands: {stats['ego']}   exo-fallback hands: {stats['fallback']}")

    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", str(PANEL / "frame_%06d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(WORK / "figure7_video_tracked.mp4")],
                   check=True, capture_output=True)
    if len(with_hand) >= 8:
        picks = [with_hand[i] for i in np.linspace(0, len(with_hand) - 1, 8).astype(int)]
        tiles = [cv2.resize(cv2.imread(str(PANEL / f"frame_{f:06d}.jpg")), (480, 480)) for f in picks]
        cv2.imwrite(str(WORK / "figure7_grid_tracked.jpg"),
                    np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    print(f"-> {WORK/'figure7_video_tracked.mp4'}, j3d_trajectory_tracked.npz")


if __name__ == "__main__":
    main()

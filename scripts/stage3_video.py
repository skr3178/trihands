#!/usr/bin/env python3
"""Steps B→C→D — full-video ego-anchored triangulation + Figure-7 panels.

For every frame:
  - rebuild the moving ego (Aria) camera (online_calibration[f] + trajectory pose),
  - for each hand the ego sees, run ego-anchored correspondence (ray-ray) + robust
    triangulation over ego + consistent exo views,
  - reproject J_3D into the ego image.

Per-frame dynamic camera selection: ego always anchors; exo views join only if they
pass the ray-ray gate that frame. Frames with no ego hand or <2 views are skipped.

Outputs: j3d_trajectory.npz, panel_frames/, figure7_video.mp4, figure7_grid.jpg.
Run with:  ../.venv-egoexo/bin/python stage3_video.py
"""
import csv, json, subprocess
from pathlib import Path
import numpy as np
import cv2

from triangulate_geom import load_gopro_cams, draw, quat_to_R, CALIB, FRAMES
from reproject_ego import build_ego_camera, TRAJ
from stage3_auto import ExoCam, EgoCam, match_exo, robust_triangulate

TRI = Path(__file__).resolve().parents[1]
KPTS_ALL = TRI / "work/kpts/all_kpts.npz"
PANEL = TRI / "work/panel_frames"
WORK = TRI / "work"
NFRAMES = 746


class Traj:
    """Fast nearest-pose lookup into the 1 kHz closed-loop trajectory."""
    def __init__(self, path):
        rows = list(csv.DictReader(open(path)))
        self.ts = np.array([int(r["tracking_timestamp_us"]) for r in rows])
        self.t = np.array([[float(r[f"t{a}_world_device"]) for a in "xyz"] for r in rows])
        self.q = np.array([[float(r[f"q{a}_world_device"]) for a in "xyzw"] for r in rows])
    def pose(self, t_us):
        i = int(np.searchsorted(self.ts, t_us))
        i = min(max(i, 0), len(self.ts) - 1)
        if i > 0 and abs(self.ts[i-1] - t_us) < abs(self.ts[i] - t_us):
            i -= 1
        return quat_to_R(*self.q[i]), self.t[i]


def dets_for(store, view, f):
    out, i = [], 0
    while f"{view}__{f:06d}__{i}__kpts2d" in store:
        out.append((store[f"{view}__{f:06d}__{i}__kpts2d"].astype(float),
                    store[f"{view}__{f:06d}__{i}__meta"]))
        i += 1
    return out


def ok(proj):
    return np.all(np.isfinite(proj)) and np.abs(proj).max() < 1e5


def main():
    exo = {v: ExoCam(c, v) for v, c in load_gopro_cams(CALIB).items()}
    kp = np.load(KPTS_ALL)
    oc = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")]
    traj = Traj(TRAJ / "closed_loop_trajectory.csv")
    PANEL.mkdir(parents=True, exist_ok=True)

    J3D_traj, with_hand = {}, []
    stats = {"L": 0, "R": 0, "skip": 0}
    for f in range(NFRAMES):
        cam, R_dc, t_dc = build_ego_camera(oc[f])
        R_wd, t_wd = traj.pose(oc[f]["tracking_timestamp_us"])
        egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd)

        ego_dets = dets_for(kp, "aria01", f)
        exo_dets = {v: [k for k, m in dets_for(kp, v, f)] for v in exo}

        img = cv2.imread(str(FRAMES / "aria01" / f"frame_{f:06d}.jpg"))
        drew = False
        for hand, want in (("R", 1), ("L", 0)):
            ego_h = next((k for k, m in ego_dets if int(m[0]) == want), None)
            if ego_h is None:
                continue
            accepted, _ = match_exo(egocam.unproject, ego_h, exo, exo_dets)
            views = [(egocam, ego_h)] + accepted
            if len(views) < 2:
                continue
            try:
                J3D, keep, med = robust_triangulate(views, ego_idx=0)
            except np.linalg.LinAlgError:
                continue
            if med.get(0, 1e9) > 40:                     # ego reproj sanity (px)
                continue
            proj = np.array([egocam.project(J3D[j]) for j in range(21)])
            if not ok(proj):
                continue
            J3D_traj[f"{f:06d}_{hand}"] = J3D
            draw(img, proj)
            stats[hand] += 1; drew = True
        if drew:
            with_hand.append(f)
        else:
            stats["skip"] += 1
        cv2.imwrite(str(PANEL / f"frame_{f:06d}.jpg"), img)
        if f % 100 == 0:
            print(f"  frame {f:4d}: L={stats['L']} R={stats['R']} skip={stats['skip']}")

    np.savez_compressed(WORK / "j3d_trajectory.npz", **J3D_traj)
    print(f"\ndone. triangulated hands: L={stats['L']}  R={stats['R']}  "
          f"frames with ≥1 hand: {len(with_hand)}/{NFRAMES}  (skipped {stats['skip']})")
    print(f"trajectory -> {WORK/'j3d_trajectory.npz'}")

    # Step D: video
    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", str(PANEL / "frame_%06d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(WORK / "figure7_video.mp4")],
                   check=True, capture_output=True)
    print(f"video -> {WORK/'figure7_video.mp4'}")

    # Step D: 2×4 grid montage of evenly-spaced frames that have a hand (Fig-7 style)
    if len(with_hand) >= 8:
        picks = [with_hand[i] for i in np.linspace(0, len(with_hand) - 1, 8).astype(int)]
        tiles = [cv2.resize(cv2.imread(str(PANEL / f"frame_{f:06d}.jpg")), (480, 480)) for f in picks]
        grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        cv2.imwrite(str(WORK / "figure7_grid.jpg"), grid)
        print(f"grid -> {WORK/'figure7_grid.jpg'}  (frames {picks})")


if __name__ == "__main__":
    main()

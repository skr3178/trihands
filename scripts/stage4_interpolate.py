#!/usr/bin/env python3
"""Stage 4 — temporal interpolation of the 3D-hand trajectory.

Fills gaps of <= 12 frames (0.4 s @ 30 fps) by LINEAR interpolation between
triangulated 3D joints (world frame), per hand. Longer gaps (e.g. the end-of-take
block where the hand leaves the ego view) are left unfilled, as the paper does.

Reprojects the augmented trajectory per-frame (the ego camera still moves) and
re-renders the Figure-7 video + grid.

Run with:  ../.venv-egoexo/bin/python stage4_interpolate.py
"""
import json, subprocess
from pathlib import Path
import numpy as np
import cv2

from triangulate_geom import draw, FRAMES
from reproject_ego import build_ego_camera, TRAJ
from stage3_auto import EgoCam
from stage3_video import Traj, NFRAMES, WORK, PANEL

MAXGAP = 12     # 0.4 s @ 30 fps


def main():
    data = np.load(WORK / "j3d_trajectory.npz")
    hands = {"R": {}, "L": {}}
    for key in data.files:
        f, h = key.split("_"); hands[h][int(f)] = data[key]

    # linear interpolation across short gaps, per hand
    interp = {"R": {}, "L": {}}
    for h in ("R", "L"):
        fr = sorted(hands[h])
        for a, b in zip(fr, fr[1:]):
            gap = b - a - 1
            if 1 <= gap <= MAXGAP:
                for f in range(a + 1, b):
                    t = (f - a) / (b - a)
                    interp[h][f] = (1 - t) * hands[h][a] + t * hands[h][b]
    print(f"interpolated frames: R +{len(interp['R'])}, L +{len(interp['L'])} (gaps ≤ {MAXGAP})")

    allhands = {h: {**hands[h], **interp[h]} for h in ("R", "L")}

    # re-render with the augmented trajectory (ego camera rebuilt per frame)
    oc = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")]
    tr = Traj(TRAJ / "closed_loop_trajectory.csv")
    augmented, with_hand = {}, []
    for f in range(NFRAMES):
        cam, R_dc, t_dc = build_ego_camera(oc[f])
        R_wd, t_wd = tr.pose(oc[f]["tracking_timestamp_us"])
        egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd)
        img = cv2.imread(str(FRAMES / "aria01" / f"frame_{f:06d}.jpg"))
        drew = False
        for h in ("R", "L"):
            if f in allhands[h]:
                J = allhands[h][f]
                proj = np.array([egocam.project(J[j]) for j in range(21)])
                draw(img, proj)
                augmented[f"{f:06d}_{h}"] = J; drew = True
        if drew:
            with_hand.append(f)
        cv2.imwrite(str(PANEL / f"frame_{f:06d}.jpg"), img)

    np.savez_compressed(WORK / "j3d_trajectory_interp.npz", **augmented)
    r_before, r_after = len(hands["R"]), len(hands["R"]) + len(interp["R"])
    print(f"right-hand frames: {r_before} -> {r_after} "
          f"({100*r_before/NFRAMES:.0f}% -> {100*r_after/NFRAMES:.0f}%)")
    print(f"frames with ≥1 hand: {len(with_hand)}/{NFRAMES} ({100*len(with_hand)/NFRAMES:.0f}%)")

    # re-render video + grid
    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", str(PANEL / "frame_%06d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(WORK / "figure7_video.mp4")],
                   check=True, capture_output=True)
    if len(with_hand) >= 8:
        picks = [with_hand[i] for i in np.linspace(0, len(with_hand) - 1, 8).astype(int)]
        tiles = [cv2.resize(cv2.imread(str(PANEL / f"frame_{f:06d}.jpg")), (480, 480)) for f in picks]
        grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        cv2.imwrite(str(WORK / "figure7_grid.jpg"), grid)
    print(f"re-rendered -> {WORK/'figure7_video.mp4'}, {WORK/'figure7_grid.jpg'}")


if __name__ == "__main__":
    main()

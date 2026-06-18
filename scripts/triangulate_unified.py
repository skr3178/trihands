#!/usr/bin/env python3
"""Stage 3 — unified triangulation over ego + exo views.

Triangulate the wearer's right hand from ALL three views (ego Aria + exo cam01 +
cam04) into one J_3D, then reproject into the ego view (Figure 7 panel). Compares
against the exo-only triangulation (Step 1) to show the accuracy gain from adding
the high-resolution ego view.

The ego view here is used as a triangulation INPUT (roles: input + reprojection
target). Correspondence is still hand-picked (wearer's right hand); auto ego-anchor
is the next step.

Run with:  ../.venv-egoexo/bin/python triangulate_unified.py
"""
import json
from pathlib import Path
import numpy as np
import cv2

from triangulate_geom import (load_gopro_cams, unproject_world, triangulate_rays,
                              draw, CALIB, FRAMES, KPTS)
from reproject_ego import build_ego_camera, world_device_pose, TRAJ, VIDEO_WH

FRAME = 300
N = VIDEO_WH
OUT = Path(__file__).resolve().parents[1] / "work/triangulation_test"


def main():
    cams = load_gopro_cams(CALIB)
    exo_kp = np.load(KPTS)
    ego_kp = np.load(OUT.parent / "kpts/ego_300.npz")

    # --- build ego camera + its world pose at this frame ---
    rec = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")][FRAME]
    ego_cam, R_dc, t_dc = build_ego_camera(rec)
    R_wd, t_wd, gap = world_device_pose(rec["tracking_timestamp_us"])
    R_wc = R_wd @ R_dc                      # T_world_camera
    t_wc = R_wd @ t_dc + t_wd

    def ego_unproject_world(px_upright):
        # undo cw90: upright (x,y) -> native (u,v) = (y, N-1-x), then unproject + to world
        u, v = px_upright[1], N - 1 - px_upright[0]
        d_cam = np.asarray(ego_cam.unproject_no_checks(np.array([u, v], float))).ravel()
        d_cam /= np.linalg.norm(d_cam)
        d_world = R_wc @ d_cam
        return t_wc, d_world / np.linalg.norm(d_world)

    def ego_project(X):
        Xc = R_wc.T @ (X - t_wc)
        px = np.asarray(ego_cam.project_no_checks(Xc)).ravel()      # native
        return np.array([N - 1 - px[1], px[0]])                     # -> upright (cw90)

    # --- correspondence: wearer's right hand ---
    exo_in = {"cam01": 0, "cam04": 1}
    exo_kpf = lambda v, i: exo_kp[f"{v}__{FRAME:06d}__{i}__kpts2d"].astype(float)
    ego_arr = ego_kp[f"aria01__{FRAME:06d}__1__kpts2d"].astype(float)   # idx1 = R

    # --- triangulate two ways: exo-only and ego+exo ---
    def triangulate(use_ego):
        J = np.zeros((21, 3))
        for j in range(21):
            O, D = [], []
            for v, i in exo_in.items():
                o, d = unproject_world(cams[v], exo_kpf(v, i)[j]); O.append(o); D.append(d)
            if use_ego:
                o, d = ego_unproject_world(ego_arr[j]); O.append(o); D.append(d)
            J[j] = triangulate_rays(O, D)
        return J

    J_exo = triangulate(use_ego=False)
    J_uni = triangulate(use_ego=True)

    # --- compare reprojection error in the EGO view vs ego's own detected keypoints ---
    for name, J in [("exo-only (2 views)", J_exo), ("ego+exo (3 views)", J_uni)]:
        proj = np.array([ego_project(J[j]) for j in range(21)])
        err = np.linalg.norm(proj - ego_arr, axis=1)
        print(f"  {name:20s}: ego reproj err vs ego detection  "
              f"mean={err.mean():5.1f}px  wrist={err[0]:5.1f}px  max={err.max():5.1f}px")

    # --- draw the unified Figure-7 panel ---
    OUT.mkdir(parents=True, exist_ok=True)
    proj = np.array([ego_project(J_uni[j]) for j in range(21)])
    img = cv2.imread(str(FRAMES / "aria01" / f"frame_{FRAME:06d}.jpg"))
    draw(img, proj, label="ego: triangulated 3D hand (ego + cam01 + cam04)")
    cv2.imwrite(str(OUT / "ego_reproj_unified.jpg"), img)
    print(f"unified Figure-7 panel -> {OUT/'ego_reproj_unified.jpg'}")


if __name__ == "__main__":
    main()

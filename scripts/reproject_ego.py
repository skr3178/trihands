#!/usr/bin/env python3
"""Stage 3 Step 2a — build the moving ego (Aria) camera and validate it.

Reproject the Step-1 triangulated hand (from exo cam01+cam04) into the egocentric
Aria image. If the skeleton lands on the wearer's hand, the ego camera pose chain
(frame -> timestamp -> world device pose -> device->camera -> fisheye) is correct,
AND the overlay is a first Figure-7 panel.

Run with:  ../.venv-egoexo/bin/python reproject_ego.py
"""
import csv, json
from pathlib import Path
import numpy as np
import cv2
from projectaria_tools.core import calibration as cal
from projectaria_tools.core.sophus import SE3

from triangulate_geom import (load_gopro_cams, unproject_world, triangulate_rays,
                              quat_to_R, draw, CALIB, FRAMES, KPTS, HAND_EDGES, EDGE_BGR)

TRI = Path(__file__).resolve().parents[1]
TRAJ = TRI / "egoexo_data/takes/sfu_cooking025_7/trajectory"
OUT = TRI / "work/triangulation_test"
FRAME = 300
NATIVE = 2880                      # Aria RGB full sensor; video is 1408 -> scale below
VIDEO_WH = 1408


def build_ego_camera(oc_rec, video_wh=VIDEO_WH, native=NATIVE):
    """Aria camera-rgb CameraCalibration rescaled to the frame-aligned video size."""
    rgb = [c for c in oc_rec["CameraCalibrations"] if c["Label"] == "camera-rgb"][0]
    p = np.array(rgb["Projection"]["Params"], dtype=np.float64)     # FISHEYE624, 15 params
    s = video_wh / native
    p = p.copy(); p[0] *= s; p[1] *= s; p[2] *= s                   # scale focal, cx, cy
    cam = cal.CameraCalibration("camera-rgb", cal.CameraModelType.FISHEYE624,
                                p, SE3(), video_wh, video_wh, None, float(np.pi), "")
    # T_Device_Camera (UnitQuaternion stored as [w, [x,y,z]])
    t = np.array(rgb["T_Device_Camera"]["Translation"])
    w, (x, y, z) = rgb["T_Device_Camera"]["UnitQuaternion"]
    R_dc = quat_to_R(x, y, z, w)
    return cam, R_dc, t


def world_device_pose(t_query_us):
    """Nearest closed_loop_trajectory pose to a timestamp (gap < 0.5 ms)."""
    best = None; bestdt = None
    for row in csv.DictReader(open(TRAJ / "closed_loop_trajectory.csv")):
        ts = int(row["tracking_timestamp_us"]); dt = abs(ts - t_query_us)
        if bestdt is None or dt < bestdt:
            bestdt = dt; best = row
        elif ts > t_query_us and dt > bestdt:
            break                                                   # sorted; past the minimum
    R_wd = quat_to_R(float(best["qx_world_device"]), float(best["qy_world_device"]),
                     float(best["qz_world_device"]), float(best["qw_world_device"]))
    t_wd = np.array([float(best["tx_world_device"]), float(best["ty_world_device"]),
                     float(best["tz_world_device"])])
    return R_wd, t_wd, bestdt


def main():
    # 1) recompute Step-1 triangulation (wearer's right hand, cam01 idx0 + cam04 idx1)
    cams = load_gopro_cams(CALIB)
    d = np.load(KPTS)
    inputs = {"cam01": 0, "cam04": 1}
    kp = lambda v, i: d[f"{v}__{FRAME:06d}__{i}__kpts2d"].astype(np.float64)
    J3D = np.zeros((21, 3))
    for j in range(21):
        o = [unproject_world(cams[v], kp(v, i)[j]) for v, i in inputs.items()]
        J3D[j] = triangulate_rays([a for a, _ in o], [b for _, b in o])

    # 2) build the ego camera for this frame
    oc = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")]
    rec = oc[FRAME]
    t_us = rec["tracking_timestamp_us"]
    ego_cam, R_dc, t_dc = build_ego_camera(rec)
    R_wd, t_wd, gap = world_device_pose(t_us)
    # compose T_world_camera = T_world_device ∘ T_device_camera
    R_wc = R_wd @ R_dc
    t_wc = R_wd @ t_dc + t_wd
    print(f"frame {FRAME}: ts={t_us}us, trajectory gap={gap}us")
    print(f"ego cam world position: {t_wc.round(3)}")

    # 3) reproject J3D into the ego image
    proj = []
    for X in J3D:
        Xc = R_wc.T @ (X - t_wc)
        px = np.asarray(ego_cam.project_no_checks(Xc)).ravel()
        proj.append(px)
    proj = np.array(proj)

    # The MPS calibration is in the native sensor orientation; the frame-aligned video is
    # upright (rotated cw90). Since the image is square, rotate the projected 2D points to match.
    N = VIDEO_WH
    proj = np.stack([N - 1 - proj[:, 1], proj[:, 0]], axis=1)        # cw90
    inb = ((proj[:, 0] >= 0) & (proj[:, 0] < N) & (proj[:, 1] >= 0) & (proj[:, 1] < N)).sum()
    print(f"reprojected wrist px (cw90): {proj[0].round(0)}  ({inb}/21 joints inside image)")

    OUT.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(FRAMES / "aria01" / f"frame_{FRAME:06d}.jpg"))
    draw(img, proj, label="ego: triangulated 3D hand (from exo cam01+cam04)")
    cv2.imwrite(str(OUT / "ego_reproj.jpg"), img)
    print(f"overlay -> {OUT/'ego_reproj.jpg'}  (a Figure-7 panel)")


if __name__ == "__main__":
    main()

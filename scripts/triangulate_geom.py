#!/usr/bin/env python3
"""Stage 3, Step 1 — geometry sanity check (exo-only, hand-picked correspondence).

Triangulate one hand's 21 keypoints from >=2 exo GoPro views using fisheye
unprojection (projectaria_tools KannalaBrandtK3) + linear ray triangulation, then
reproject the 3D result into every exo view to confirm the camera math is correct.

No trajectory, no correspondence solver — correspondence is hand-supplied so that
ONLY the fisheye-unproject + DLT geometry is under test.

Run with:  ../.venv-egoexo/bin/python triangulate_geom.py
"""
import csv
from pathlib import Path
import numpy as np
import cv2
from projectaria_tools.core import calibration as cal
from projectaria_tools.core.sophus import SE3

TRI = Path(__file__).resolve().parents[1]                       # trihands/
CALIB = TRI / "egoexo_data/takes/sfu_cooking025_7/trajectory/gopro_calibs.csv"
FRAMES = TRI / "work/frames"
KPTS = TRI / "work/kpts/exo_probe.npz"
OUT = TRI / "work/triangulation_test"

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
              (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
              (0,17),(17,18),(18,19),(19,20)]
EDGE_BGR = [c for c in [(0,0,255),(0,165,255),(0,255,255),(0,255,0),(255,0,0)] for _ in range(4)]


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def load_gopro_cams(csv_path):
    cams = {}
    for row in csv.DictReader(open(csv_path)):
        # projectaria KannalaBrandtK3 layout = [fx, fy, cx, cy, k0, k1, k2, k3] (8 params)
        params = np.array([float(row[f"intrinsics_{i}"]) for i in range(8)], dtype=np.float64)
        W, H = int(row["image_width"]), int(row["image_height"])
        camcal = cal.CameraCalibration(
            row["cam_uid"], cal.CameraModelType.KANNALA_BRANDT_K3,
            params, SE3(), W, H, None, float(np.pi), "")          # T_Device_Camera = identity
        R_wc = quat_to_R(float(row["qx_world_cam"]), float(row["qy_world_cam"]),
                         float(row["qz_world_cam"]), float(row["qw_world_cam"]))
        t_wc = np.array([float(row["tx_world_cam"]), float(row["ty_world_cam"]),
                         float(row["tz_world_cam"])])
        cams[row["cam_uid"]] = dict(cal=camcal, R_wc=R_wc, t_wc=t_wc, WH=(W, H))
    return cams


def unproject_world(cam, pixel):
    """2D pixel -> (ray origin, unit ray direction) in world frame."""
    d_cam = np.asarray(cam["cal"].unproject_no_checks(np.asarray(pixel, float))).ravel()
    d_cam /= np.linalg.norm(d_cam)
    d_world = cam["R_wc"] @ d_cam
    return cam["t_wc"], d_world / np.linalg.norm(d_world)


def project_world(cam, X):
    """world 3D point -> 2D pixel."""
    X_cam = cam["R_wc"].T @ (X - cam["t_wc"])
    px = cam["cal"].project_no_checks(np.asarray(X_cam, float))
    return np.asarray(px).ravel()


def triangulate_rays(origins, dirs):
    """Least-squares 3D point closest to a set of rays (o_i, d_i)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)        # projector onto plane perp to d
        A += P; b += P @ o
    return np.linalg.solve(A, b)


def draw(img, kpts, color_box=None, label=None):
    for (a, b), col in zip(HAND_EDGES, EDGE_BGR):
        cv2.line(img, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)), col, 3)
    for p in kpts:
        cv2.circle(img, tuple(p.astype(int)), 4, (255, 255, 255), -1)
    if label:
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return img


def main():
    cams = load_gopro_cams(CALIB)
    d = np.load(KPTS)
    frame = 300
    # hand-picked correspondence: wearer's RIGHT hand
    # (cam01 idx0, cam04 idx1 are the triangulation inputs; cam02 idx0 is the held-out check)
    inputs = {"cam01": 0, "cam04": 1}
    holdout = {"cam02": 0}

    def kp(view, idx):
        return d[f"{view}__{frame:06d}__{idx}__kpts2d"].astype(np.float64)  # (21,2)

    # triangulate each of the 21 joints from the input views
    J3D = np.zeros((21, 3))
    for j in range(21):
        origins, dirs = [], []
        for view, idx in inputs.items():
            o, dr = unproject_world(cams[view], kp(view, idx)[j])
            origins.append(o); dirs.append(dr)
        J3D[j] = triangulate_rays(origins, dirs)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== triangulated wrist (joint 0) world coord: {J3D[0].round(3)} ===")

    # reproject into every view, report error vs detected keypoints, draw overlays
    all_views = {**inputs, **holdout}
    for view, idx in all_views.items():
        proj = np.array([project_world(cams[view], J3D[j]) for j in range(21)])
        det = kp(view, idx)
        err = np.linalg.norm(proj - det, axis=1)
        tag = "INPUT" if view in inputs else "HELD-OUT"
        print(f"  {view} [{tag}] reproj err vs detection: "
              f"mean={err.mean():.1f}px  wrist={err[0]:.1f}px  max={err.max():.1f}px")
        img = cv2.imread(str(FRAMES / view / f"frame_{frame:06d}.jpg"))
        draw(img, proj, label=f"{view} {tag}: reprojected 3D")
        cv2.imwrite(str(OUT / f"{view}_reproj.jpg"), img)
    print(f"overlays -> {OUT}")


if __name__ == "__main__":
    main()

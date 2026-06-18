#!/usr/bin/env python3
"""Reproduce Figure 7 — our triangulated hands (top) vs EgoExo4D GT (bottom).

On a hand-annotated take (iiith_cooking_111_4), for each annotated frame:
  - OUR pipeline: WiLoR + ego-anchored triangulation -> 3D hand (top row, "Ours")
  - EgoExo4D's released hand annotation (3D, world frame, partial) -> bottom row (GT)
Both projected into the ego view. Reports 3D MPJPE (ours vs GT, mm) over the joints
EgoExo4D annotated.

Run with:  ../.venv-egoexo/bin/python compare_gt.py
"""
import json
from pathlib import Path
import numpy as np
import cv2

from triangulate_geom import load_gopro_cams, draw, HAND_EDGES, EDGE_BGR
from reproject_ego import build_ego_camera
from stage3_auto import ExoCam, EgoCam, match_exo, robust_triangulate
from stage3_video import Traj, dets_for

TRI = Path(__file__).resolve().parents[1]
UID = "83167883-277a-4720-accb-4aeb0253e60e"
TD = TRI / "egoexo_data/takes/iiith_cooking_111_4"
TRAJ = TD / "trajectory"
CALIB = TRAJ / "gopro_calibs.csv"
FRAMES = TRI / "work/frames_iiith"
KPTS = TRI / "work/kpts/iiith_kpts.npz"
ANN = TRI / f"egoexo_data/annotations/ego_pose/train/hand/annotation/{UID}.json"
WORK = TRI / "work"

# MANO 21-joint order -> EgoExo4D joint names (right hand)
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
MANO_NAMES = ["right_wrist"] + [f"right_{fg}_{i}" for fg in FINGERS for i in (1, 2, 3, 4)]


def gt_hand(ann_frame):
    a3d = ann_frame[0].get("annotation3D", {})
    J = np.full((21, 3), np.nan)
    for i, name in enumerate(MANO_NAMES):
        if name in a3d:
            J[i] = [a3d[name]["x"], a3d[name]["y"], a3d[name]["z"]]
    return J


def draw_partial(img, pts2d, valid):
    for (a, b), col in zip(HAND_EDGES, EDGE_BGR):
        if valid[a] and valid[b]:
            cv2.line(img, tuple(pts2d[a].astype(int)), tuple(pts2d[b].astype(int)), col, 2)
    for i, p in enumerate(pts2d):
        if valid[i]:
            cv2.circle(img, tuple(p.astype(int)), 3, (255, 255, 255), -1)
    return img


def main():
    exo = {v: ExoCam(c, v) for v, c in load_gopro_cams(CALIB).items()}
    oc = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")]
    traj = Traj(TRAJ / "closed_loop_trajectory.csv")
    ann = json.load(open(ANN))
    kp = np.load(KPTS)
    frames = sorted(int(k) for k in ann.keys())

    mpjpe, panels = [], []
    for f in frames:
        if f >= len(oc):
            continue
        cam, R_dc, t_dc = build_ego_camera(oc[f])
        R_wd, t_wd = traj.pose(oc[f]["tracking_timestamp_us"])
        egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd)

        gt = gt_hand(ann[str(f)])
        gt_valid = ~np.isnan(gt).any(axis=1)

        # our ego-anchored triangulation
        ego_dets = dets_for(kp, "aria01", f)
        exo_dets = {v: [k for k, m in dets_for(kp, v, f)] for v in exo}
        ego_R = next((k for k, m in ego_dets if int(m[0]) == 1), None)
        ours = None
        if ego_R is not None:
            acc, _ = match_exo(egocam.unproject, ego_R, exo, exo_dets)
            views = [(egocam, ego_R)] + acc
            if len(views) >= 2:
                try:
                    J, _, med = robust_triangulate(views, ego_idx=0)
                    if med.get(0, 1e9) <= 40:
                        ours = J
                except np.linalg.LinAlgError:
                    pass

        if ours is not None and gt_valid.sum() >= 5:
            e = np.linalg.norm(ours[gt_valid] - gt[gt_valid], axis=1) * 1000  # mm
            mpjpe.append((f, e.mean(), int(gt_valid.sum())))

        panels.append((f, egocam, ours, gt, gt_valid))

    # ---- metric ----
    if mpjpe:
        errs = np.array([m for _, m, _ in mpjpe])
        print(f"=== Ours vs EgoExo4D GT — 3D MPJPE over annotated joints ===")
        print(f"  frames compared: {len(mpjpe)}")
        print(f"  MPJPE: mean={errs.mean():.1f}mm  median={np.median(errs):.1f}mm  "
              f"min={errs.min():.1f}  max={errs.max():.1f}")

    # ---- Figure-7 two-row grid: top=Ours, bottom=GT, on frames with dense GT + ours ----
    good = [p for p in panels if p[2] is not None and p[4].sum() >= 15]
    good = good or [p for p in panels if p[2] is not None and p[4].sum() >= 10]
    picks = [good[i] for i in np.linspace(0, len(good) - 1, min(5, len(good))).astype(int)]
    top, bot = [], []
    for f, egocam, ours, gt, gv in picks:
        img_o = cv2.imread(str(FRAMES / "aria01" / f"frame_{f:06d}.jpg"))
        img_g = img_o.copy()
        draw(img_o, np.array([egocam.project(ours[j]) for j in range(21)]))
        gp = np.array([egocam.project(gt[j]) if gv[j] else np.zeros(2) for j in range(21)])
        draw_partial(img_g, gp, gv)
        cv2.putText(img_o, "Ours", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(img_g, "EgoExo4D GT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 3)
        top.append(cv2.resize(img_o, (480, 480))); bot.append(cv2.resize(img_g, (480, 480)))
    grid = np.vstack([np.hstack(top), np.hstack(bot)])
    cv2.imwrite(str(WORK / "figure7_compare.jpg"), grid)
    print(f"two-row comparison -> {WORK/'figure7_compare.jpg'}  (frames {[p[0] for p in picks]})")


if __name__ == "__main__":
    main()

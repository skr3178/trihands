#!/usr/bin/env python3
"""Figure-7 comparison on frames where the LEFT hand has >=3 views (the accurate
ones). Contrast against figure7_compare_full.jpg (which includes noisy 2-view left
hands). Per-frame left-hand MPJPE is labelled so the improvement is visible.

Writes a NEW file: work/figure7_compare_3view.jpg (overwrites nothing).
Run with:  ../.venv-egoexo/bin/python compare_gt_3view.py
"""
import json
import numpy as np
import cv2

from triangulate_geom import load_gopro_cams, draw
from reproject_ego import build_ego_camera
from stage3_auto import ExoCam, EgoCam, match_exo, robust_triangulate
from stage3_video import Traj, dets_for
import compare_gt_full as C

MIN_LEFT_VIEWS = 3


def main():
    exo = {v: ExoCam(c, v) for v, c in load_gopro_cams(C.CALIB).items()}
    oc = [json.loads(l) for l in open(C.TRAJ / "online_calibration.jsonl")]
    traj = Traj(C.TRAJ / "closed_loop_trajectory.csv")
    ann = json.load(open(C.ANN)); kp = np.load(C.KPTS)
    gt_frames = sorted(int(k) for k in ann.keys())

    recs = {}
    for f in gt_frames:
        if f >= len(oc):
            continue
        cam, R_dc, t_dc = build_ego_camera(oc[f])
        R_wd, t_wd = traj.pose(oc[f]["tracking_timestamp_us"])
        egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd)
        ego_dets = dets_for(kp, "aria01", f)
        exo_dets = {v: [k for k, m in dets_for(kp, v, f)] for v in exo}
        rec = {"ego": egocam}
        for hand, want in (("R", 1), ("L", 0)):
            ego_h = next((k for k, m in ego_dets if int(m[0]) == want), None)
            if ego_h is None:
                continue
            acc, _ = match_exo(egocam.unproject, ego_h, exo, exo_dets)
            views = [(egocam, ego_h)] + acc
            if len(views) < 2:
                continue
            try:
                J, _, med = robust_triangulate(views, ego_idx=0)
            except np.linalg.LinAlgError:
                continue
            if med.get(0, 1e9) > 40:
                continue
            rec[hand] = J; rec["n" + hand] = len(views)
        recs[f] = rec

    gv = lambda f, hand: ~np.isnan(C.gt_hand(ann[str(f)], hand)).any(1)
    good = [f for f in gt_frames
            if "L" in recs.get(f, {}) and recs[f].get("nL", 0) >= MIN_LEFT_VIEWS
            and gv(f, "L").sum() >= 8 and "R" in recs[f]]
    print(f"frames where LEFT hand has >={MIN_LEFT_VIEWS} views (+dense GT): {len(good)}")
    picks = [good[i] for i in np.linspace(0, len(good) - 1, min(5, len(good))).astype(int)]

    top, bot = [], []
    for f in picks:
        ego = recs[f]["ego"]
        io = cv2.imread(str(C.FRAMES / "aria01" / f"frame_{f:06d}.jpg")); ig = io.copy()
        gt = C.gt_hand(ann[str(f)], "L"); v = gv(f, "L")
        eL = np.linalg.norm(recs[f]["L"][v] - gt[v], axis=1).mean() * 1000
        for hand in ("R", "L"):
            if hand in recs[f]:
                draw(io, np.array([ego.project(recs[f][hand][j]) for j in range(21)]))
            g = C.gt_hand(ann[str(f)], hand); vv = ~np.isnan(g).any(1)
            gp = np.array([ego.project(g[j]) if vv[j] else np.zeros(2) for j in range(21)])
            C.draw_partial(ig, gp, vv)
        cv2.putText(io, f"Ours  left: {recs[f]['nL']} views, {eL:.0f}mm",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 3)
        cv2.putText(ig, "EgoExo4D GT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
        top.append(cv2.resize(io, (480, 480))); bot.append(cv2.resize(ig, (480, 480)))
    cv2.imwrite(str(C.WORK / "figure7_compare_3view.jpg"),
                np.vstack([np.hstack(top), np.hstack(bot)]))
    print(f"picked frames {picks}")
    print(f"-> {C.WORK/'figure7_compare_3view.jpg'}")


if __name__ == "__main__":
    main()

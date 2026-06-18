#!/usr/bin/env python3
"""Why is the left hand worse? — accuracy vs camera count, and the >=3-views fix.

Stratifies 3D MPJPE (vs EgoExo4D GT) by the number of cameras used, per hand, on
iiith_cooking_111_4. Shows the left hand only fails at 2 views (no redundancy to
absorb noisier left-hand keypoints), and that requiring >=3 views fixes it.

Produces a LICENSE-SAFE chart (mpjpe_by_views.png) — pure data, no dataset imagery.
Run with:  ../.venv-egoexo/bin/python analyze_views.py
"""
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from triangulate_geom import load_gopro_cams
from reproject_ego import build_ego_camera
from stage3_auto import ExoCam, EgoCam, match_exo, robust_triangulate
from stage3_video import Traj, dets_for
import compare_gt_full as C


def main():
    exo = {v: ExoCam(c, v) for v, c in load_gopro_cams(C.CALIB).items()}
    oc = [json.loads(l) for l in open(C.TRAJ / "online_calibration.jsonl")]
    traj = Traj(C.TRAJ / "closed_loop_trajectory.csv")
    ann = json.load(open(C.ANN)); kp = np.load(C.KPTS)
    gt_frames = sorted(int(k) for k in ann.keys())

    data = {}
    for hand, want in (("R", 1), ("L", 0)):
        by_nv = defaultdict(list)
        for f in gt_frames:
            if f >= len(oc):
                continue
            cam, R_dc, t_dc = build_ego_camera(oc[f])
            R_wd, t_wd = traj.pose(oc[f]["tracking_timestamp_us"])
            egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd)
            ego_dets = dets_for(kp, "aria01", f)
            exo_dets = {v: [k for k, m in dets_for(kp, v, f)] for v in exo}
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
            gt = C.gt_hand(ann[str(f)], hand); v = ~np.isnan(gt).any(1)
            if v.sum() < 5:
                continue
            by_nv[len(views)].append(np.linalg.norm(J[v] - gt[v], axis=1).mean() * 1000)
        data[hand] = by_nv

    # ---- print summary + the >=3-views fix ----
    print("=== 3D MPJPE (mm) vs EgoExo4D GT, by camera count ===")
    for hand in ("R", "L"):
        allf = [e for es in data[hand].values() for e in es]
        ge3 = [e for nv, es in data[hand].items() if nv >= 3 for e in es]
        print(f"  {hand}: all-views mean={np.mean(allf):.0f}mm (n={len(allf)})  |  "
              f">=3 views mean={np.mean(ge3):.0f}mm (n={len(ge3)})")

    # ---- chart ----
    nvs = [2, 3, 4]
    x = np.arange(len(nvs)); w = 0.38
    rm = [np.mean(data["R"].get(nv, [np.nan])) for nv in nvs]
    lm = [np.mean(data["L"].get(nv, [np.nan])) for nv in nvs]
    rn = [len(data["R"].get(nv, [])) for nv in nvs]
    ln = [len(data["L"].get(nv, [])) for nv in nvs]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    b1 = ax.bar(x - w / 2, rm, w, label="Right hand", color="#1f77b4")
    b2 = ax.bar(x + w / 2, lm, w, label="Left hand", color="#d62728")
    for b, n in zip(b1, rn):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}mm\nn={n}",
                ha="center", va="bottom", fontsize=8)
    for b, n in zip(b2, ln):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}mm\nn={n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{n} views" for n in nvs])
    ax.set_ylabel("3D MPJPE vs EgoExo4D GT (mm)")
    ax.set_title("Hand triangulation accuracy by camera count (take iiith_cooking_111_4)\n"
                 "Left hand fails ONLY at 2 views — no redundancy to absorb noisier left-hand keypoints")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(lm) * 1.25)
    fig.tight_layout()
    fig.savefig(C.WORK / "mpjpe_by_views.png", dpi=120)
    print(f"chart -> {C.WORK/'mpjpe_by_views.png'}")


if __name__ == "__main__":
    main()

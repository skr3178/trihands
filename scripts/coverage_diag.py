#!/usr/bin/env python3
"""Frame-wise camera-coverage diagnostic for the right-hand triangulation.

For every frame, records which cameras contributed (ego anchor + exo views that
passed the ray-ray gate) and why a frame was skipped. Produces:
  - coverage_plot.png : (top) #cameras used per frame, (bottom) camera×frame heatmap
  - a printed summary of skip reasons + per-camera usage.

Reuses the cached keypoints (work/kpts/all_kpts.npz) — no WiLoR rerun.
Run with:  ../.venv-egoexo/bin/python coverage_diag.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from triangulate_geom import load_gopro_cams, CALIB
from reproject_ego import build_ego_camera, TRAJ
from stage3_auto import ExoCam, EgoCam, match_exo, robust_triangulate
from stage3_video import Traj, dets_for, KPTS_ALL, NFRAMES, WORK

CAMS = ["ego", "cam01", "cam02", "cam03", "cam04"]
STATUS_COLOR = {"ok": "#2ca02c", "sanity": "#ff7f0e", "few_views": "#d62728", "no_ego": "#7f7f7f"}


def main():
    exo = {v: ExoCam(c, v) for v, c in load_gopro_cams(CALIB).items()}
    kp = np.load(KPTS_ALL)
    oc = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")]
    traj = Traj(TRAJ / "closed_loop_trajectory.csv")

    contrib = np.zeros((5, NFRAMES))          # which cameras contributed (R hand)
    status = []
    for f in range(NFRAMES):
        cam, R_dc, t_dc = build_ego_camera(oc[f])
        R_wd, t_wd = traj.pose(oc[f]["tracking_timestamp_us"])
        egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd)
        ego_dets = dets_for(kp, "aria01", f)
        exo_dets = {v: [k for k, m in dets_for(kp, v, f)] for v in exo}

        ego_h = next((k for k, m in ego_dets if int(m[0]) == 1), None)   # right hand
        if ego_h is None:
            status.append("no_ego"); continue
        contrib[0, f] = 1
        accepted, _ = match_exo(egocam.unproject, ego_h, exo, exo_dets)
        for camobj, _ in accepted:
            contrib[CAMS.index(camobj.name), f] = 1
        views = [(egocam, ego_h)] + accepted
        if len(views) < 2:
            status.append("few_views"); continue
        try:
            _, _, med = robust_triangulate(views, ego_idx=0)
        except np.linalg.LinAlgError:
            status.append("sanity"); continue
        status.append("sanity" if med.get(0, 1e9) > 40 else "ok")

    status = np.array(status)
    ncam = contrib.sum(axis=0)

    # ---- summary ----
    print("=== right-hand coverage summary ===")
    for s in ("ok", "sanity", "few_views", "no_ego"):
        print(f"  {s:10s}: {(status == s).sum():4d} frames")
    print("  per-camera usage (frames contributing):")
    for i, name in enumerate(CAMS):
        print(f"    {name}: {int(contrib[i].sum())}")
    print(f"  mean cameras/frame (triangulated only): "
          f"{ncam[status=='ok'].mean():.2f}")

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), height_ratios=[1, 1.1], sharex=True)
    colors = [STATUS_COLOR[s] for s in status]
    ax1.bar(range(NFRAMES), ncam, color=colors, width=1.0)
    ax1.set_ylabel("# cameras used\n(R hand)")
    ax1.set_ylim(0, 5.3)
    ax1.set_title("Per-frame camera coverage — right-hand triangulation (take sfu_cooking025_7)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLOR.values()]
    ax1.legend(handles, [f"{k} ({(status==k).sum()})" for k in STATUS_COLOR],
               ncol=4, loc="upper right", fontsize=8)

    ax2.imshow(contrib, aspect="auto", cmap="Greens", interpolation="nearest",
               extent=[0, NFRAMES, 4.5, -0.5])
    ax2.set_yticks(range(5)); ax2.set_yticklabels(CAMS)
    ax2.set_ylabel("camera"); ax2.set_xlabel("frame index")
    ax2.set_title("Which cameras contributed each frame (green = used)")
    fig.tight_layout()
    out = WORK / "coverage_plot.png"
    fig.savefig(out, dpi=110)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()

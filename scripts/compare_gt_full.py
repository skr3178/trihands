#!/usr/bin/env python3
"""Full-pipeline Figure-7 comparison vs EgoExo4D GT — BOTH hands.

Runs the complete pipeline over the continuous annotated window (653-953):
ego-anchored triangulation + temporal exo-fallback + Stage-4 interpolation, for
left AND right hands, then compares against EgoExo4D's released hand annotations on
the GT frames (3D MPJPE).

Writes only NEW files (figure7_compare_full.jpg) — overwrites nothing.
Needs work/kpts/iiith_kpts_full.npz (WiLoR over the whole window).
Run with:  ../.venv-egoexo/bin/python compare_gt_full.py
"""
import json
from pathlib import Path
import numpy as np
import cv2

from triangulate_geom import load_gopro_cams, draw, HAND_EDGES, EDGE_BGR
from reproject_ego import build_ego_camera
from stage3_auto import ExoCam, EgoCam, match_exo, robust_triangulate
from stage3_tracked import match_exo_pred, MAX_TRACK
from stage3_video import Traj, dets_for

TRI = Path(__file__).resolve().parents[1]
UID = "83167883-277a-4720-accb-4aeb0253e60e"
TD = TRI / "egoexo_data/takes/iiith_cooking_111_4"
TRAJ = TD / "trajectory"
CALIB = TRAJ / "gopro_calibs.csv"
FRAMES = TRI / "work/frames_iiith"
KPTS = TRI / "work/kpts/iiith_kpts_full.npz"
ANN = TRI / f"egoexo_data/annotations/ego_pose/train/hand/annotation/{UID}.json"
WORK = TRI / "work"
WIN = range(653, 954)
MAXGAP = 12

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
GT_NAMES = {"R": ["right_wrist"] + [f"right_{fg}_{i}" for fg in FINGERS for i in (1, 2, 3, 4)],
            "L": ["left_wrist"] + [f"left_{fg}_{i}" for fg in FINGERS for i in (1, 2, 3, 4)]}


def gt_hand(ann_frame, hand):
    a3d = ann_frame[0].get("annotation3D", {})
    J = np.full((21, 3), np.nan)
    for i, n in enumerate(GT_NAMES[hand]):
        if n in a3d:
            J[i] = [a3d[n]["x"], a3d[n]["y"], a3d[n]["z"]]
    return J


def draw_partial(img, pts, valid):
    for (a, b), c in zip(HAND_EDGES, EDGE_BGR):
        if valid[a] and valid[b]:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), c, 2)
    for i, p in enumerate(pts):
        if valid[i]:
            cv2.circle(img, tuple(p.astype(int)), 3, (255, 255, 255), -1)


def main():
    exo = {v: ExoCam(c, v) for v, c in load_gopro_cams(CALIB).items()}
    oc = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")]
    traj = Traj(TRAJ / "closed_loop_trajectory.csv")
    ann = json.load(open(ANN))
    kp = np.load(KPTS)
    gt_frames = sorted(int(k) for k in ann.keys())

    # ---- full tracked pipeline over the window, both hands ----
    ours = {"R": {}, "L": {}}
    src = {"R": {}, "L": {}}                       # how each frame was obtained
    last = {"R": None, "L": None}; last_f = {"R": -999, "L": -999}
    egocams = {}
    for f in WIN:
        if f >= len(oc):
            continue
        cam, R_dc, t_dc = build_ego_camera(oc[f])
        R_wd, t_wd = traj.pose(oc[f]["tracking_timestamp_us"])
        egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd); egocams[f] = egocam
        ego_dets = dets_for(kp, "aria01", f)
        exo_dets = {v: [k for k, m in dets_for(kp, v, f)] for v in exo}
        for hand, want in (("R", 1), ("L", 0)):
            J, how = None, None
            ego_h = next((k for k, m in ego_dets if int(m[0]) == want), None)
            if ego_h is not None:
                acc, _ = match_exo(egocam.unproject, ego_h, exo, exo_dets)
                views = [(egocam, ego_h)] + acc
                if len(views) >= 2:
                    try:
                        Jc, _, med = robust_triangulate(views, ego_idx=0)
                        if med.get(0, 1e9) <= 40:
                            J, how = Jc, "ego"
                    except np.linalg.LinAlgError:
                        pass
            if J is None and last[hand] is not None and (f - last_f[hand]) <= MAX_TRACK:
                acc = match_exo_pred(last[hand], exo, exo_dets)
                if len(acc) >= 2:
                    try:
                        Jc, _, med = robust_triangulate(acc, ego_idx=None)
                        if np.median(list(med.values())) <= max(c.tau for c, _ in acc):
                            J, how = Jc, "fallback"
                    except np.linalg.LinAlgError:
                        pass
            if J is not None:
                ours[hand][f] = J; src[hand][f] = how; last[hand] = J; last_f[hand] = f

    # ---- Stage 4 interpolation (gaps <= 12 frames) ----
    for hand in ("R", "L"):
        fr = sorted(ours[hand])
        for a, b in zip(fr, fr[1:]):
            if 1 <= b - a - 1 <= MAXGAP:
                for f in range(a + 1, b):
                    t = (f - a) / (b - a)
                    ours[hand][f] = (1 - t) * ours[hand][a] + t * ours[hand][b]
                    src[hand][f] = "interp"

    # ---- compare vs GT on annotated frames, both hands ----
    print("=== Full pipeline (ego-anchored + exo-fallback + interp) vs EgoExo4D GT ===")
    for hand in ("R", "L"):
        errs = []
        for f in gt_frames:
            if f not in ours[hand]:
                continue
            gt = gt_hand(ann[str(f)], hand); v = ~np.isnan(gt).any(1)
            if v.sum() < 5:
                continue
            errs.append(np.linalg.norm(ours[hand][f][v] - gt[v], axis=1).mean() * 1000)
        cov = sum(1 for f in gt_frames if f in ours[hand])
        nego = sum(1 for f in gt_frames if src[hand].get(f) == "ego")
        nfb = sum(1 for f in gt_frames if src[hand].get(f) == "fallback")
        nint = sum(1 for f in gt_frames if src[hand].get(f) == "interp")
        e = np.array(errs)
        tag = f"mean={e.mean():.1f}mm median={np.median(e):.1f}mm" if len(e) else "n/a"
        print(f"  {hand} hand: covers {cov}/{len(gt_frames)} GT frames "
              f"(ego={nego} fallback={nfb} interp={nint}); MPJPE on {len(e)}: {tag}")

    # ---- two-row grid, both hands ----
    picks = [f for f in gt_frames if f in ours["R"]
             and (~np.isnan(gt_hand(ann[str(f)], "R")).any(1)).sum() >= 12]
    picks = [picks[i] for i in np.linspace(0, len(picks) - 1, min(5, len(picks))).astype(int)]
    top, bot = [], []
    for f in picks:
        ego = egocams[f]
        io = cv2.imread(str(FRAMES / "aria01" / f"frame_{f:06d}.jpg")); ig = io.copy()
        for hand in ("R", "L"):
            if f in ours[hand]:
                draw(io, np.array([ego.project(ours[hand][f][j]) for j in range(21)]))
            gt = gt_hand(ann[str(f)], hand); v = ~np.isnan(gt).any(1)
            gp = np.array([ego.project(gt[j]) if v[j] else np.zeros(2) for j in range(21)])
            draw_partial(ig, gp, v)
        cv2.putText(io, "Ours L+R", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(ig, "EgoExo4D GT L+R", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
        top.append(cv2.resize(io, (480, 480))); bot.append(cv2.resize(ig, (480, 480)))
    cv2.imwrite(str(WORK / "figure7_compare_full.jpg"),
                np.vstack([np.hstack(top), np.hstack(bot)]))
    print(f"-> {WORK/'figure7_compare_full.jpg'} (frames {picks})")


if __name__ == "__main__":
    main()

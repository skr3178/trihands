#!/usr/bin/env python3
"""Stage 3 — automatic ego-anchored triangulation (lines 9, 15, 18, 22).

For one frame:
  1. ego anchor: the ego view sees only the wearer (clean L/R). For each ego hand,
     match each exo view's detection by ray-consistency (triangulate ego+candidate,
     reproject into the exo view); accept the best if < τ_c, else reject (bystander).
  2. robust triangulate: IRLS Huber reweighting over ego + accepted exo views, drop
     any view whose reproj error > τ_c (keep ego).
  3. reproject J_3D into the ego view (Figure 7 panel).

Validation target (frame 300): auto-pick cam01+cam04 for the wearer's right hand,
REJECT cam04's bystander detections, reproduce the hand-picked unified result.

Run with:  ../.venv-egoexo/bin/python stage3_auto.py
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


# ---- uniform camera interface: pixel(upright) <-> world ray / pixel ----
class ExoCam:
    def __init__(self, c, name):
        self.c, self.name = c, name
        self.tau = 0.01 * max(c["WH"])
    def unproject(self, px):  return unproject_world(self.c, px)
    def project(self, X):     return project_world_exo(self.c, X)

def project_world_exo(c, X):
    Xc = c["R_wc"].T @ (X - c["t_wc"])
    return np.asarray(c["cal"].project_no_checks(np.asarray(Xc, float))).ravel()


class EgoCam:
    """Aria camera at one frame; handles the native<->upright cw90 rotation."""
    def __init__(self, cam, R_wc, t_wc):
        self.cam, self.R_wc, self.t_wc = cam, R_wc, t_wc
        self.tau = 0.01 * N
    def unproject(self, px_upright):
        u, v = px_upright[1], N - 1 - px_upright[0]                  # upright -> native
        d = np.asarray(self.cam.unproject_no_checks(np.array([u, v], float))).ravel()
        d /= np.linalg.norm(d)
        dw = self.R_wc @ d
        return self.t_wc, dw / np.linalg.norm(dw)
    def project(self, X):
        Xc = self.R_wc.T @ (X - self.t_wc)
        px = np.asarray(self.cam.project_no_checks(Xc)).ravel()      # native
        return np.array([N - 1 - px[1], px[0]])                     # -> upright


MATCH_M = 0.03          # ego-anchor ray-ray consistency threshold (3 cm)


def wtriangulate(O, D, w):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d, wi in zip(O, D, w):
        P = wi * (np.eye(3) - np.outer(d, d))
        A += P; b += P @ o
    X, *_ = np.linalg.lstsq(A, b, rcond=None)   # robust to near-singular (degenerate baseline)
    return X


def ray_ray_dist(o1, d1, o2, d2):
    """Closest distance between two 3D rays (resolution-independent, in meters)."""
    n = np.cross(d1, d2); nn = np.linalg.norm(n)
    if nn < 1e-9:
        return float(np.linalg.norm(np.cross(o2 - o1, d1)))
    return float(abs(np.dot(o2 - o1, n)) / nn)


def ray_perp(X, o, d):
    v = X - o
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def match_exo(ego_unproj, ego_kpts, exo_cams, exo_dets):
    """Ego anchor: pick each exo view's detection that is ray-consistent with the
    ego hand (mean ray-ray distance < MATCH_M). Resolution-independent → rejects
    noisy views, wrong-hand, and bystanders alike."""
    ego_rays = [ego_unproj(ego_kpts[j]) for j in range(21)]
    accepted, log = [], []
    for name, cam in exo_cams.items():
        best, best_d = np.inf, None
        for d in exo_dets.get(name, []):
            dist = np.mean([ray_ray_dist(*ego_rays[j], *cam.unproject(d[j])) for j in range(21)])
            if dist < best: best, best_d = dist, d
        if best_d is not None and best < MATCH_M:
            accepted.append((cam, best_d)); log.append(f"{name}:✓({best*100:.1f}cm)")
        else:
            log.append(f"{name}:✗({best*100:.1f}cm)")
    return accepted, log


def robust_triangulate(views, ego_idx=0):
    """IRLS Huber over views, drop views with reproj err > τ_c (keep ego_idx).

    Residuals are normalized by each view's τ_c so exo (4K) and ego (1408) mix
    correctly in the Huber weighting.
    """
    J = np.zeros((21, 3))
    keep = list(range(len(views)))
    rays = [[v[0].unproject(v[1][j]) for j in range(21)] for v in views]
    taus = [v[0].tau for v in views]
    med = {}
    for _ in range(3):                                             # outer: drop bad views
        for j in range(21):
            O = [rays[i][j][0] for i in keep]; D = [rays[i][j][1] for i in keep]
            w = np.ones(len(keep))
            for _ in range(5):                                     # IRLS Huber on RAY residuals
                X = wtriangulate(O, D, w)
                res = np.array([ray_perp(X, O[k], D[k]) for k in range(len(keep))])  # meters
                delta = 0.02                                       # 2 cm — resolution-independent
                w = np.where(res <= delta, 1.0, delta / np.maximum(res, 1e-9))
            J[j] = X
        med = {i: float(np.median([np.linalg.norm(views[i][0].project(J[j]) - views[i][1][j])
                                   for j in range(21)])) for i in keep}
        bad = [i for i in keep if i != ego_idx and med[i] > taus[i]]
        if not bad: break
        keep.remove(max(bad, key=lambda i: med[i]))
    return J, keep, med


def main():
    exo = {v: ExoCam(c, v) for v, c in load_gopro_cams(CALIB).items()}
    exo_kp = np.load(KPTS); ego_kp = np.load(OUT.parent / "kpts/ego_300.npz")

    # build ego camera at this frame
    rec = [json.loads(l) for l in open(TRAJ / "online_calibration.jsonl")][FRAME]
    cam, R_dc, t_dc = build_ego_camera(rec)
    R_wd, t_wd, _ = world_device_pose(rec["tracking_timestamp_us"])
    egocam = EgoCam(cam, R_wd @ R_dc, R_wd @ t_dc + t_wd)

    # gather detections
    def dets(store, view):
        out, i = [], 0
        while f"{view}__{FRAME:06d}__{i}__kpts2d" in store:
            out.append((store[f"{view}__{FRAME:06d}__{i}__kpts2d"].astype(float),
                        store[f"{view}__{FRAME:06d}__{i}__meta"]))
            i += 1
        return out
    exo_dets_all = {v: [k for k, m in dets(exo_kp, v)] for v in exo}
    ego_dets = dets(ego_kp, "aria01")

    # process the wearer's RIGHT hand (ego idx with meta[0]==1)
    ego_R = next(k for k, m in ego_dets if m[0] == 1)
    print(f"frame {FRAME}: ego anchor = right hand; matching exo views...")
    accepted, log = match_exo(egocam.unproject, ego_R, exo, exo_dets_all)
    print("  " + "  ".join(log))

    views = [(egocam, ego_R)] + accepted
    J3D, keep, med = robust_triangulate(views, ego_idx=0)
    names = ["ego"] + [a[0].name for a in accepted]
    kept_names = [names[i] for i in keep]
    print(f"  triangulated from: {kept_names}")
    print("  per-view median reproj err: " +
          "  ".join(f"{names[i]}={med[i]:.0f}px(τ{views[i][0].tau:.0f})" for i in keep))

    proj = np.array([egocam.project(J3D[j]) for j in range(21)])
    err = np.linalg.norm(proj - ego_R, axis=1)
    print(f"  ego reproj err vs ego detection: mean={err.mean():.1f}px max={err.max():.1f}px")
    OUT.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(FRAMES / "aria01" / f"frame_{FRAME:06d}.jpg"))
    draw(img, proj, label=f"ego: AUTO ego-anchored ({'+'.join(kept_names)})")
    cv2.imwrite(str(OUT / "ego_reproj_auto.jpg"), img)
    print(f"panel -> {OUT/'ego_reproj_auto.jpg'}")


if __name__ == "__main__":
    main()

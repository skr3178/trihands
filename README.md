# trihands

Reproduction of the **multiview hand triangulation pipeline** (Algorithm 1, Supplementary B) from
*"What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos?"*
([arXiv:2606.06627](https://arxiv.org/abs/2606.06627), Li et al.).

The pipeline turns EgoExo4D multi-view RGB video (1 egocentric Aria camera + 4 exocentric GoPros,
all fisheye) into **3D hand keypoints**, then reprojects them into the egocentric view to produce a
**Figure-7-style panel**.

## Status

| Stage | What it does | Status |
|---|---|---|
| 1 — Hand detection | YOLO bounding boxes per view | ✅ |
| 2 — Single-view WiLoR | 2D hand keypoints per view | ✅ |
| 3 — Multiview triangulation | ego-anchored correspondence + fisheye-unproject + DLT + IRLS/τ_c | ✅ |
| — Ego reprojection | project 3D hand into the ego image (= Figure 7) | ✅ single hand |
| 4–5 | temporal interpolation / IK→MANO | skipped (not needed for the keypoint panel) |

See [`algorithm.md`](algorithm.md) for the annotated algorithm + notation, and [`goal.md`](goal.md)
for the plan, findings, and build log.

## How it works (Stage 3 highlights)

- **Fisheye, not pinhole.** Exo cameras are KannalaBrandtK3, the ego Aria is FisheyeRadTanThinPrism.
  2D keypoints are unprojected to world-frame rays via `projectaria_tools`; DLT triangulates on rays.
- **Ego-anchored correspondence.** The ego view sees only the wearer, so it anchors which detections
  are the wearer's hands. Each exo detection is matched by **ray-ray distance** (resolution-independent):
  the wearer's hand rays intersect at ~0.5 cm, wrong-hand at ~10 cm, bystanders at ~22 cm — so noisy
  views, wrong hands, and bystanders are rejected automatically.
- **Robust triangulation.** IRLS Huber on ray-perpendicular residuals + τ_c view-drop.

## Scripts

| Script | Role |
|---|---|
| `scripts/wilor_keypoints.py` | Stage 1–2: WiLoR detection + 2D keypoints per view (runs in a WiLoR env) |
| `scripts/triangulate_geom.py` | Stage 3 geometry: exo fisheye-unproject + ray-DLT |
| `scripts/reproject_ego.py` | builds the moving ego (Aria) camera; reprojects 3D into the ego view |
| `scripts/triangulate_unified.py` | triangulate from ego + exo together |
| `scripts/stage3_auto.py` | **full automatic Stage 3**: ego-anchored correspondence + robust triangulation |

## Data (not included)

This repo contains **only code and docs**. Not included (and `.gitignore`d):

- **EgoExo4D** video + calibration — license forbids redistribution; download via the official
  [Ego-Exo4D downloader](https://docs.ego-exo4d-data.org/) with your own license.
- Extracted frames, keypoint caches, and result overlays (regenerable from the above).
- Model checkpoints (WiLoR / MANO) — see their respective repos.

The figures under `figures/` are from the source paper, included for reference.

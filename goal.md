# Goal — Multiview Hand Triangulation (TriHands reproduction)

Reproduce the **multiview hand triangulation pipeline** (Algorithm 1, Supplementary B) from
*"What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos?"*
(`figures/Hand Pose Estimation.pdf`).

**Concrete target:** a **Figure-7-like panel** — an egocentric (Aria) frame with the hand
skeleton overlaid, where that skeleton is the **3D hand triangulated from the multi-view
cameras and reprojected into the ego view** (not a monocular estimate). Figure 8 / 9 (the same
triangulation reprojected into the exo views) is the intermediate visual.

> This supersedes the earlier monocular RLDX-1 goal (preserved in `../4.2.md`,
> `../setup_log.md`, `../monocular_capture_pipeline_plan.md`, and `../samples/`).

## Input data — EgoExo4D (multi-view, calibrated)

- One egocentric **Aria** RGB camera (moving, head-worn) + **4 exo GoPros** (static), all
  frame-aligned at 30 fps, all **fisheye**.
- Per-camera intrinsics + extrinsics provided; the ego camera's pose comes from a ~1 kHz SLAM
  trajectory. All cameras share **one world frame** (matching `graph_uid`) — this is what makes
  triangulation possible.
- On disk: take `sfu_cooking025_7` (25 s, 746 frames/view) under `egoexo_data/takes/`. Access
  via the licensed `egoexo` CLI (creds in `~/.aws`, expire Jul 2 2026). 678 cooking takes exist.

## Pipeline

Full algorithm + notation: **`algorithm.md`**. Stage status toward the Figure-7 target:

| Stage | Role | Needed for Fig 7? | Status |
|---|---|---|---|
| 1 — Hand detection | YOLO bbox per view | ✅ | **done** — `scripts/wilor_keypoints.py` |
| 2 — Single-view WiLoR | 2D keypoints `J_2D` per view | ✅ | **done** — `scripts/wilor_keypoints.py` |
| 3 — Multiview triangulation | unproject → DLT → robust refine → subset/τ_c select → `J_3D` | ✅ **the core** | **DONE & automatic** (`stage3_auto.py`): ego-anchored ray-ray correspondence + IRLS Huber/τ_c |
| 4 — Temporal interpolation | fill gaps ≤12 frames | ✅ | `stage4_interpolate.py` — linear; R 90%→93% |
| 5 — IK → MANO `θ` | fit mesh params | ❌ (keypoints, not mesh) | skip |
| Ego reprojection (Supp C viz) | project `J_3D` into ego image | ✅ | **done (full video)** — `scripts/stage3_video.py` → `figure7_video.mp4`, `figure7_grid.jpg` |

## Key findings / risks

- **Fisheye, not pinhole.** Algorithm 1's `P_c = K_c[I|0]T_c⁻¹` is post-undistortion shorthand.
  We **unproject the 2D keypoints to rays** with `projectaria_tools` (point-level, not whole
  images), then DLT on rays. See `algorithm.md` › *Implementation notes*.
- **Bystanders break exo-only triangulation.** EgoExo4D scenes contain secondary people whose
  hands the exo detector also fires on. Confirmed on `work/frames/cam04/frame_000300.jpg`: the
  wearer is central (both hands detected at the cutting board) but **two bystanders** stand in
  the kitchen and the detector fires on one's hands too — see the overlay
  `work/overlays_exo_probe/cam04/frame_000300.jpg` (extra `R 0.45 / L 0.40` detections on the
  green-shirt person at right). This is exactly what the paper calls out (Supp B):

  > "EgoExo4D videos often contain secondary individuals near the ego-camera wearer. To avoid
  > their hands, we require the selected camera set to include valid hand detections for the
  > egocentric view."

  So the paper's **ego anchor** (`c_ego ∈ S`) + reprojection gate is what identifies the
  *wearer's* hands. Pure exo-only is ambiguous; the ego view is needed for correspondence, not
  just for the final Fig-7 reprojection.
- **Uneven exo coverage.** cam01/cam04 see the wearer well (conf 0.6–0.8); cam02 sparse, cam03
  mostly empty. Best wearer hand visibility is at the cutting-board frames (~300–450), not the
  side-counter frames.

## Build order

1. **Stage 3 geometry sanity check** — hand-pick the wearer's right hand in 2–3 exo views at a
   good frame (e.g. frame 300: cam01 + cam04), fisheye-unproject + DLT → 3D, reproject back into
   those exo views and confirm it lands on the hand. *No trajectory, no correspondence solver —
   isolates the camera math.*
2. **Ego-anchored correspondence** — add the ego view (needs the trajectory) so the subset/τ_c
   gate auto-selects the wearer's hands and rejects bystanders. The paper's actual Stage 3.
3. **Ego reprojection → Figure 7** — project `J_3D` into the ego image and overlay. ✅
4. **Full-video loop** ✅ — `scripts/stage3_video.py` runs the per-frame ego-anchored pipeline over
   all 746 frames: **671 right + 508 left hands triangulated, 686/746 frames (92%)**, outputs the
   3D-hand trajectory (`j3d_trajectory.npz`) + `figure7_video.mp4` + `figure7_grid.jpg`. Per-frame
   dynamic camera selection (ego anchors; exo views join via the ray-ray gate). `lstsq` used for
   degenerate-baseline frames; frames w/o ego hand or <2 views are skipped.
5. **Temporal exo-fallback** ✅ (extension beyond Algorithm 1) — `scripts/stage3_tracked.py`. When
   the ego camera loses the hand, propagate its 3D from the last good frame, reproject into the exo
   views, match the nearest detection (bystanders are far → rejected), and triangulate **exo-only**;
   a reproj-consistency check stops divergence. Reclaims the multi-view redundancy the strict ego
   anchor discarded: **right-hand coverage 93% → 100%** (1177 ego-anchored + 287 exo-fallback hands).
   Diagnosis (`coverage_diag.py`): all 54 ego-miss frames had ≥2 exo views seeing a hand — none were
   truly hidden, so the 93% "ceiling" was a method limit, not a data limit. Outputs:
   `j3d_trajectory_tracked.npz`, `figure7_video_tracked.mp4`, `figure7_grid_tracked.jpg`.
6. *(optional)* EgoExo4D-GT comparison row (Fig 7 bottom) = released hand annotations projected into
   ego — **blocked for sfu_cooking025_7 (no hand annotations; only camera_pose)**; needs a
   hand-annotated take. More takes; Stage 5 (MANO mesh).

## Documentation

### Fisheye → pinhole conversion

Triangulation needs straight 3D rays from each camera; fisheye lenses bend those rays, so the
distortion must be undone first. **We convert the 2D keypoints (the ~21 points), not the
images.** WiLoR already produced keypoint pixels on the raw fisheye frames — those pixels need
no conversion. The conversion happens *after*, per point:

```
fisheye image ──WiLoR──▶ 2D keypoint pixels        (done, on fisheye image — no conversion)
                              │
                              ▼  ← THE CONVERSION (per point, via projectaria_tools)
2D pixel ──fisheye unproject──▶ 3D bearing ray (a "straightened" direction in the world frame)
                              │
   rays from ≥2 cameras ──DLT──▶ 3D point  J_3D
                              │
   3D point ──fisheye project──▶ pixel   (to check reprojection error / draw overlays)
```

- **Unproject the points, don't undistort the images.** We have only 21 points per hand —
  converting those is exact, instant, and lossless. Undistorting whole 4K images is expensive,
  discards the wide-FOV edges (fisheye sees >180° that won't fit a pinhole frame), and would
  force re-running detection. No benefit for triangulation.
- The unprojection uses each camera's true fisheye model via `projectaria_tools`
  (exo = KannalaBrandtK3, ego = FisheyeRadTanThinPrism). After unprojection, DLT triangulates on
  the straightened rays exactly as if the cameras were pinhole.
- Algorithm 1's `P_c = K_c[I|0]T_c⁻¹` (pinhole) is therefore valid only *after* this conversion;
  the refinement/error operator `π_c` uses the full fisheye projection. The paper does the same
  (undistort fisheye → pinhole, Sec 4.2, ref [28] Kannala–Brandt).

### Step 1 — Geometry sanity check (no trajectory, no correspondence solver) ✅ DONE

Hand-pick the wearer's right hand in 2–3 exo views at a good frame (e.g. frame 300:
cam01 R0.64 + cam04 R0.64), triangulate via fisheye-unproject + DLT, then reproject the 3D point
back into those exo views and check it lands on the hand. This proves the camera math works —
the single biggest unknown. It deliberately avoids the trajectory and the multi-person
correspondence problem so that *only* the fisheye DLT is under test.

**Result (`scripts/triangulate_geom.py`):** triangulated the wearer's right hand from cam01 +
cam04; reprojection error = **2.8 px** (cam01) / **7.2 px** (cam04) on the input views, and the
3D hand lands on the wearer's hand in the **held-out cam02** (44 px mean on a 4K frame) — see
`work/triangulation_test/cam02_reproj.jpg`. Fisheye-unproject → ray-DLT → fisheye-reproject is
geometrically correct.

> Gotcha fixed: projectaria's `KANNALA_BRANDT_K3` takes **8 params** `[fx, fy, cx, cy, k0..k3]`,
> not 7 — a 7-param vector is silently misread (focal/principal-point shifted), giving ~700 px error.

### Step 2 — Ego-anchored correspondence

Add the ego view (needs the trajectory) so the pipeline auto-selects the wearer's hands and
rejects bystanders via the reprojection-error gate — the paper's actual method
(`c_ego ∈ S` + the τ_c threshold). The ego camera only ever sees the wearer's own hands, so it
anchors which exo detections are the wearer's.

### Step 3 — Figure 7 reprojection into the ego view  🔨 (ego camera validated)

Project the triangulated `J_3D` into the ego image and overlay the skeleton — the Figure-7
panel.

**Ego camera works (`scripts/reproject_ego.py`):** the Step-1 hand (triangulated from exo
cam01+cam04) reprojects onto the wearer's hand in ego frame 300 — see
`work/triangulation_test/ego_reproj.jpg`. Pose chain: `frame i → online_calibration[i]` (timestamp
+ FisheyeRadTanThinPrism intrinsics + `T_Device_Camera`) → nearest pose in `closed_loop_trajectory`
→ `T_world_camera`. Remaining for a full panel: run across frames + the ego-anchored correspondence
(Step 2) so it's automatic, not hand-picked.

> Gotcha fixed: MPS ego intrinsics are in the **native sensor orientation**, but the frame-aligned
> video is **upright**. `rotate_camera_calib_cw90deg` is Linear-model-only, so instead we project in
> native orientation then rotate the 2D points **cw90** (valid because the image is square 1408²).
> Also: ego intrinsics are for the ~2880² native sensor → scale focal/cx/cy by 1408/2880 = 0.489.

This order confirms the fisheye DLT is correct **before** layering on the harder multi-person
correspondence, so each step is verifiable in isolation.

## Environments

- `.venv-hamer` — WiLoR + HaMeR (torch 2.8/cu128, CUDA). Runs Stages 1–2.
- `.venv-egoexo` — `egoexo` downloader + `projectaria_tools` (fisheye geometry, Aria pose). No
  torch. Runs Stage 3 geometry.
- Frames: `work/frames/<view>/frame_%06d.jpg`; keypoints/overlays under `work/` (all gitignored).

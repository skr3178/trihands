# Algorithm 1 — Multiview Hand Triangulation Pipeline

Reference for the triangulation algorithm we are reproducing. Source: **"What Matters
When Cotraining Robot Manipulation Policies on Everyday Human Videos?"**, Supplementary B,
page 13 (`figures/Hand Pose Estimation.pdf`; the boxed pseudocode is also in
`figures/multiview_hand_pipeline.png`).

The pipeline turns **multi-view RGB video + camera calibration** into **3D hand keypoints**
(and optionally MANO parameters). The 3D hands are EgoExo4D's high-quality "ground-truth"
labels; a Figure-7 panel is those 3D hands projected back into the egocentric image.

---

## Notation

### Indices and sets
| Symbol | Meaning |
|---|---|
| `f` | frame index (videos are frame-aligned across cameras, 30 fps) |
| `c` | a camera **and** its associated image at frame `f` (the paper overloads the symbol) |
| `c_ego` | the egocentric camera (Aria RGB); the exo cameras are the static GoPros |
| `h` | a hand, `h ∈ {L, R}` (left / right) |
| `j` | a hand joint index, `j = 1 … 21` (MANO 21-keypoint layout) |
| `C` | the set of cameras that have a valid detection of hand `h` at frame `f` |
| `S` | a candidate **subset** of cameras used for one triangulation; `S ⊆ C` |
| `S*` | the chosen best subset for that frame/hand |

### Inputs (Require)
| Symbol | Meaning | EgoExo4D source on disk |
|---|---|---|
| `V = {V_ego, V_exo1 … V_exoN}` | multi-view video streams | `frame_aligned_videos/aria01_214-1.mp4` (ego) + `cam01–04.mp4` (exo) |
| `K_c` | 3×3 **intrinsic** matrix of camera `c` | exo: `gopro_calibs.csv`; ego: `online_calibration.jsonl` (`camera-rgb`) |
| `T_c` | 4×4 **extrinsic** SE(3) pose of camera `c` (camera→world) | exo: static, `gopro_calibs.csv`; ego: time-varying `T_c(f)` from `closed_loop_trajectory.csv` ∘ `T_Device_Camera` |

> Both `K_c` and `T_c` describe **fisheye** cameras here — see *Implementation notes*.

### Intermediate / output data
| Symbol | Meaning |
|---|---|
| `B_f^c` | bounding box(es) of hands in frame `f`, camera `c` |
| `I_patch` | cropped hand image patch fed to the single-view network |
| `J_2D[c, j]` | 2D pixel location of joint `j` in camera `c` (one hand) — **triangulation input** |
| `J_3D[h, f]` | triangulated 3D position of the 21 joints of hand `h` at frame `f` — **the output** |
| `X_j` | the triangulated 3D position of a single joint `j` (a column of `J_3D`) |
| `θ_init` | per-view MANO parameters predicted by the single-view network (used to init IK) |
| `θ[h, f]` | final MANO pose parameters for hand `h` at frame `f` (Stage 5 output) |

### Operators and constants
| Symbol | Meaning |
|---|---|
| `P_c = K_c [I \| 0] T_c⁻¹` | 3×4 **projection matrix**: world point → pixel (pinhole form; valid after undistortion) |
| `[I \| 0]` | 3×4 matrix `[identity \| zero column]` (drops the homogeneous 4th coord) |
| `T_c⁻¹` | world→camera transform (inverse of the camera→world pose) |
| `DLT(·)` | Direct Linear Transformation: closed-form linear triangulation from ≥2 views |
| `π_c(X)` | **projection operator** — projects 3D point `X` into camera `c`'s image (general; may carry the full fisheye model) |
| `ρ_H(r)` | **Huber loss**, robust to outliers: `½r²` if `\|r\| ≤ δ`, else `δ\|r\| − ½δ²` |
| `δ` | Huber transition threshold (quadratic ↔ linear) |
| `ε_j[c]` | reprojection error of joint `j` in camera `c`: `‖π_c(X_j) − J_2D[c, j]‖` |
| `τ_c` | per-camera reprojection threshold = `0.01 · max(H_c, W_c)` (1% of the larger image dim) |
| `SLERP(·, g_max)` | temporal interpolation across gaps up to `g_max = 12` frames (0.4 s @ 30 fps) |
| `M(θ)` | MANO forward model: pose params `θ` → 3D joint positions |

---

## Pipeline at a glance

Status:  ✅ done   🔨 in progress   ⬜ to do   ⏭️ skipped (not needed for Figure 7)

```
 INPUT    5 fisheye videos: 1 ego (Aria) + 4 exo (GoPro)  +  calib {K_c , T_c}
 OUTPUT   J_3D  =  21 hand-joint 3D positions, per frame, per hand
═══════════════════════════════════════════════════════════════════════════════

 PER VIEW (×5) · PER FRAME
 ┌─────────────────────────────────────────────────────────────────┐
 │  STAGE 1   detect hands       YOLO   ─▶  bounding boxes  B_f^c    │  ✅
 │  STAGE 2   reconstruct hand   WiLoR  ─▶  2D keypoints    J_2D     │  ✅
 └────────────────────────────────┬────────────────────────────────┘
                                  │   J_2D   (21 px / hand / view)
                                  ▼
 FUSE ALL VIEWS · PER FRAME · PER HAND
 ┌─────────────────────────────────────────────────────────────────┐
 │  STAGE 3   multi-view triangulation                              │
 │                                                                  │
 │   ┌── correspondence ──────────────────────────────────────┐    │
 │   │   which detection in each view = the same hand?         │    │  ✅
 │   │   ego anchor + ray-ray gate  (rejects bystanders)       │    │
 │   └─────────────────────────────────────────────────────────┘    │
 │   ┌── per joint  j = 1 … 21 ───────────────────────────────┐    │
 │   │   pixel  ──fisheye unproject──▶  world ray              │    │  ✅ ego+exo
 │   │   ≥2 rays  ──DLT──▶  X_j   (3D point)                   │    │  ✅
 │   │   IRLS Huber refine  +  drop views over τ_c             │    │  ✅
 │   └─────────────────────────────────────────────────────────┘    │
 └────────────────────────────────┬────────────────────────────────┘
                                  │   J_3D
                                  ▼
 ┌── STAGE 4  interpolate gaps ≤ 12 frames ──┐   ✅  linear (stage4_interpolate.py)
 ├── STAGE 5  inverse kinematics → MANO θ ───┤   ⏭️  skip (keypoints, not mesh)
 └────────────────────────────────────────────┘
                                  │
                                  ▼   Supp C viz (not in Algorithm 1)
 reproject J_3D into the ego image  ─▶  FIGURE 7 panel          ✅ single hand
```

Scripts: Stage 1–2 = `scripts/wilor_keypoints.py` · Stage 3 (geometry) =
`triangulate_geom.py` (exo) + `reproject_ego.py` (ego) + `triangulate_unified.py` (ego+exo) ·
**full auto Stage 3 = `stage3_auto.py`** (ego-anchored correspondence via ray-ray distance +
IRLS Huber/τ_c, validated on frame 300: auto-selects ego+cam01+cam04, rejects cam02/bystanders,
8.3px). Remaining: batch keypoints over all frames → loop → render the multi-panel figure.

---

## Exact paper pseudocode (reference)

The verbatim Algorithm 1 (Supp B, p.13), for fidelity. Our fisheye/ego adaptations and status
are shown in the diagram above; line numbers match the paper.

```
Require: multi-view video V, intrinsics {K_c}, extrinsics {T_c} for each camera c
Ensure:  3D keypoints J_3D, MANO parameters θ

  // Stage 1 — Hand Detection
1   for all frame f, camera c:
2       B_f^c ← HANDDETECTOR(V_c[f])                         ▷ bounding boxes

  // Stage 2 — Single-View 3D Estimation
4   for all frame f, camera c, hand h ∈ {L,R}:
5       I_patch ← CROP(V_c[f], B_f^c[h])
6       J_2D, θ_init ← WiLoR(I_patch)                        ▷ 2D joints & MANO

  // Stage 3 — Multiview Triangulation
8   for all frame f, hand h:
9       C ← { c : B_f^c[h] exists }
10      if |C| < 2: continue
12      for all c ∈ C:  P_c ← K_c [I|0] T_c⁻¹
15      for all subset S ⊆ C with c_ego ∈ S, |S| ≥ 2:
16          for all joint j = 1 … 21:
17              X_j ← DLT({ P_c, J_2D[c,j] }_{c∈S})
18              X_j ← arg min_X Σ ρ_H(π_c(X) − J_2D[c,j])
19              ε_j[c] ← ‖ π_c(X_j) − J_2D[c,j] ‖
22      S* ← arg max_{|S|}  s.t.  ≥95% joints have ε_j[c] < τ_c
23      J_3D[h,f] ← X[S*]

  // Stage 4 — Temporal Interpolation
25  J_3D ← SLERP(J_3D, g_max = 12)

  // Stage 5 — Inverse Kinematics
26  for all frame f, hand h:
27      θ_0  ← θ_init[c_ego, h, f]                           ▷ HaWoR init
28      J_tgt ← T_cam⁻¹ J_3D[h,f]
29      θ[h,f] ← arg min_θ ‖ M(θ) − J_tgt ‖²
31  return J_3D, θ
```

---

## Plain-English walkthrough

**Stage 1 — Hand Detection.** Run a hand detector on every frame of every camera to get
bounding boxes `B_f^c`. (Paper note: the default Ultralytics YOLO had high false-negative
and chirality-error rates on EgoExo4D, so they swapped in a Hands23 model fine-tuned on
EpicKitchens.)

**Stage 2 — Single-View 3D Estimation.** Crop each detected hand and run WiLoR on the patch.
We keep its **2D keypoints `J_2D`** (the triangulation input) and its MANO prediction
`θ_init` (only used later to initialize IK). This is per-view and per-frame — no 3D fusion yet.

**Stage 3 — Multiview Triangulation (the core).** For each (frame, hand):
1. Collect the cameras `C` that detected this hand. Need ≥2 to triangulate.
2. Build each camera's projection matrix `P_c` from its intrinsics and extrinsics.
3. Search over camera **subsets** `S` (each must include the ego camera, `c_ego ∈ S`):
   - Triangulate every joint with **DLT** for a fast linear initial estimate.
   - **Refine** each joint by minimizing the **Huber-robust reprojection error** across the
     subset (down-weights outlier 2D detections).
   - Record each camera's reprojection error `ε_j[c]`.
4. **Select** the subset `S*` that uses the *most* cameras while still keeping ≥95% of joints
   under the threshold `τ_c`. Its triangulated joints become `J_3D[h, f]`.

   *Why the ego anchor and subset search matter:* EgoExo4D scenes contain bystanders, and
   exo detections are noisy. Requiring `c_ego ∈ S` (the ego camera only sees the wearer's own
   hands) plus the reprojection gate is what rejects bystander hands and bad views.

**Stage 4 — Temporal Interpolation.** Fill short tracking gaps (≤12 frames = 0.4 s) by
interpolation. *(Note: line 25 says `SLERP`, but the Supp B prose says "0.4 s **linear**
interpolation between 3D joints" — an internal inconsistency in the paper. For 3D joint
**positions**, linear interpolation is the correct primitive.)*

**Stage 5 — Inverse Kinematics.** Convert triangulated joints into MANO parameters `θ` by
fitting `M(θ)` to the 3D targets, **initialized from HaWoR's** MANO prediction (more reliable
in metric scale than WiLoR's). Optimization is GPU-batched. *This stage produces the hand
**mesh** parameters; it is not needed if you only want keypoints.*

---

## Implementation notes (our reproduction)

**Fisheye correction.** EgoExo4D cameras are fisheye (exo = KannalaBrandtK3, ego =
FisheyeRadTanThinPrism), so the pinhole `P_c = K_c[I|0]T_c⁻¹` on line 13 is only valid **after
undistortion**. The paper undistorts fisheye → pinhole (Sec 4.2, ref [28] Kannala–Brandt). We
implement this by **unprojecting each 2D pixel to a world-frame bearing ray** with
`projectaria_tools`, then triangulating on rays; the refinement/error operator `π_c` uses the
true fisheye projection. The rest of Stage 3 (DLT, Huber, subset/τ_c selection, ego anchor) is
unchanged.

**What we implement for a Figure-7 panel.** Figure 7 shows **keypoints**, i.e. `J_3D`
reprojected into the ego image. So:

| Stage | Needed for Figure 7? | Status |
|---|---|---|
| 1 — Detection | ✅ | done (WiLoR YOLO, `scripts/wilor_keypoints.py`) |
| 2 — Single-view `J_2D` | ✅ | done (`scripts/wilor_keypoints.py`) |
| 3 — Triangulation | ✅ **core to build** | TODO |
| 4 — Interpolation | ❌ (per-frame stills) | skip |
| 5 — IK → MANO `θ` | ❌ (keypoints, not mesh) | skip |
| *Ego reprojection (Supp C, not in Alg. 1)* | ✅ | TODO — needs ego `T_c(f)` from trajectory |

The ego reprojection that actually *draws* Figure 7 is a **visualization step** (paper Supp C,
"we visualize 2D projections in the egocamera of our triangulations in Fig 8, 9"), not part of
Algorithm 1, which ends at `J_3D`.

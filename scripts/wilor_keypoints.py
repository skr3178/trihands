#!/usr/bin/env python3
"""Stage 2 — per-view WiLoR hand keypoints.

Runs WiLoR's YOLO hand detector + 3D reconstruction on frames, and saves the
21 hand-joint 2D keypoints in *full-image pixel* coordinates (plus handedness,
detection confidence, and 3D joints in the hand frame) to an .npz.

These 2D pixels are the input to multi-view triangulation (Stage 3): they are
real image coordinates, so they can later be unprojected through each camera's
true fisheye model regardless of WiLoR's internal weak-perspective assumption.

Run with the WiLoR env:  ../.venv-hamer/bin/python wilor_keypoints.py ...
"""
import sys, os, argparse
from pathlib import Path
import numpy as np
import cv2
import torch

ORIG_CWD = Path.cwd()                    # capture before chdir; CLI paths resolve against this
WILOR_DIR = Path(__file__).resolve().parents[2] / "WiLoR"
os.chdir(WILOR_DIR)                      # WiLoR expects ./pretrained_models relative paths
sys.path.insert(0, str(WILOR_DIR))

# torch>=2.6 defaults weights_only=True, which old ultralytics ckpts can't load
_torch_load = torch.load
def _load_full(*a, **k):
    k.setdefault("weights_only", False)
    return _torch_load(*a, **k)
torch.load = _load_full

from wilor.models import load_wilor                       # noqa: E402
from wilor.utils import recursive_to                      # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset   # noqa: E402
from wilor.utils.renderer import cam_crop_to_full         # noqa: E402
from ultralytics import YOLO                              # noqa: E402

# MANO/WiLoR 21-joint hand skeleton: wrist(0) + 5 fingers (thumb..pinky)
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),        (0,5),(5,6),(6,7),(7,8),
              (0,9),(9,10),(10,11),(11,12),   (0,13),(13,14),(14,15),(15,16),
              (0,17),(17,18),(18,19),(19,20)]
FINGER_BGR = [(0,0,255),(0,165,255),(0,255,255),(0,255,0),(255,0,0)]  # thumb..pinky
EDGE_BGR   = [c for c in FINGER_BGR for _ in range(4)]


def project_full_img(points, cam_t, focal, img_res):
    """WiLoR weak-perspective projection of 3D joints to full-image pixels."""
    cx, cy = img_res[0] / 2.0, img_res[1] / 2.0
    K = np.eye(3); K[0,0] = K[1,1] = focal; K[0,2] = cx; K[1,2] = cy
    p = points + cam_t
    p = p / p[..., -1:]
    return (K @ p.T).T[..., :2]


def load_models(device):
    model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                            cfg_path="./pretrained_models/model_config.yaml")
    detector = YOLO("./pretrained_models/detector.pt")
    model = model.to(device).eval()
    detector = detector.to(device)
    return model, cfg, detector


def run_image(img_cv2, model, cfg, detector, device, conf=0.3, rescale=2.0):
    det = detector(img_cv2, conf=conf, verbose=False)[0]
    boxes, right, dconf = [], [], []
    for d in det:
        b = d.boxes.data.cpu().numpy().squeeze()
        boxes.append(b[:4]); right.append(d.boxes.cls.cpu().item())
        dconf.append(float(d.boxes.conf.cpu().item()))
    if not boxes:
        return []
    boxes = np.stack(boxes); right = np.array(right)
    ds = ViTDetDataset(cfg, img_cv2, boxes, right, rescale_factor=rescale)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    results, bi = [], 0
    for batch in dl:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        mult = (2 * batch["right"] - 1)
        pred_cam = out["pred_cam"]; pred_cam[:, 1] = mult * pred_cam[:, 1]
        bc = batch["box_center"].float(); bs = batch["box_size"].float()
        isz = batch["img_size"].float()
        sf = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * isz.max()
        camt = cam_crop_to_full(pred_cam, bc, bs, isz, sf).cpu().numpy()
        for n in range(batch["img"].shape[0]):
            joints = out["pred_keypoints_3d"][n].cpu().numpy()
            isr = int(batch["right"][n].cpu().numpy())
            joints[:, 0] = (2 * isr - 1) * joints[:, 0]
            kpts2d = project_full_img(joints, camt[n], float(sf), isz[n].cpu().numpy())
            results.append(dict(kpts2d=kpts2d.astype(np.float32),
                                joints3d=joints.astype(np.float32),
                                cam_t=camt[n].astype(np.float32),
                                is_right=isr, det_conf=float(dconf[bi]),
                                box=boxes[bi].astype(np.float32)))
            bi += 1
    return results


def draw(img, results):
    for r in results:
        k = r["kpts2d"]
        for (a, b), col in zip(HAND_EDGES, EDGE_BGR):
            cv2.line(img, tuple(k[a].astype(int)), tuple(k[b].astype(int)), col, 2)
        for p in k:
            cv2.circle(img, tuple(p.astype(int)), 3, (255, 255, 255), -1)
        x0, y0 = r["box"][:2].astype(int)
        cv2.putText(img, f"{'R' if r['is_right'] else 'L'} {r['det_conf']:.2f}",
                    (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True, help="dir with <view>/frame_%06d.jpg")
    ap.add_argument("--views", nargs="+", default=["aria01", "cam01", "cam02", "cam03", "cam04"])
    ap.add_argument("--frames", nargs="+", type=int, default=None, help="explicit frame indices")
    ap.add_argument("--frame_range", nargs=2, type=int, default=None, help="START END (inclusive)")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--draw_dir", default=None, help="if set, save skeleton overlays here")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    frames = args.frames if args.frames is not None else \
        list(range(args.frame_range[0], args.frame_range[1] + 1))
    args.frames = frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, detector = load_models(device)
    resolve = lambda p: (p if Path(p).is_absolute() else ORIG_CWD / p)
    root = resolve(args.frames_root)
    args.out_npz = resolve(args.out_npz)
    if args.draw_dir:
        args.draw_dir = resolve(args.draw_dir)
    store = {}
    for view in args.views:
        for f in args.frames:
            p = root / view / f"frame_{f:06d}.jpg"
            if not p.exists():
                print(f"  missing {p}"); continue
            img = cv2.imread(str(p))
            res = run_image(img, model, cfg, detector, device, conf=args.conf)
            for hi, r in enumerate(res):
                key = f"{view}__{f:06d}__{hi}"
                store[key + "__kpts2d"] = r["kpts2d"]
                store[key + "__joints3d"] = r["joints3d"]
                store[key + "__meta"] = np.array(
                    [r["is_right"], r["det_conf"], *r["box"]], dtype=np.float32)
            tags = ", ".join(("R" if r["is_right"] else "L") + f"{r['det_conf']:.2f}" for r in res)
            print(f"  {view} f{f:06d}: {len(res)} hand(s) [{tags}]")
            if args.draw_dir:
                od = Path(args.draw_dir) / view
                od.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(od / f"frame_{f:06d}.jpg"), draw(img, res))
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **store)
    print(f"saved {len(store)//3} hand detections -> {args.out_npz}")


if __name__ == "__main__":
    main()

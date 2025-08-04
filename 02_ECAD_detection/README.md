# ECAD Diagram Understanding: Object Detection + Connectivity

End-to-end system for understanding electronic schematics:
1) **Synthetic data generator** (components + augmentations),
2) **Object detection** (YOLOv8 / Faster R-CNN),
3) **Connectivity reconstruction** between components (classical CV → connection matrix).

> **Stack:** Python, PyTorch, YOLOv8 / Faster R-CNN (ResNet-50), OpenCV, Roboflow-format datasets

---

## At a glance
- **Goal:** detect schematic components (R/C/L/diode/transistor/…) and recover **who is connected to whom**.
- **Why it’s interesting:** blends **deep detection** with **graph extraction** from images (topology, not just boxes).
- **What I built:** data generator, training/eval for detectors, and a classical-CV connectivity module with metrics.

![sample detections](docs/img/detections.png)
*Detections with class labels and scores (placeholder).*

---

## Architecture

```text
[Data Generator] ---> [COCO-like Dataset]
       |
       |-- augmentations: affine/perspective, flips, blur, noise, illumination
       `-- synthetic wires placed consistently with component geometry

[Object Detector]
   |-- YOLOv8 (anchor-free)
   `-- Faster R-CNN (ResNet-50 + RPN)

[Connectivity Analysis]
   |-- brightness normalization + histogram equalization
   |-- thresholding, morphology, Canny edges
   |-- contours + polyline approximation
   `-- build NxN connection matrix from wire/box intersections
```

## Data

- Hand-drawn component glyphs (~60 per class) + backgrounds.
- Generator composes scenes, writes YOLO/COCO labels, and injects random wires that avoid crossing components.
- Final splits: train/val/test ≈ 300/300/300 synthetic images each (extendable with real schematics).
- Key augmentations: affine/perspective, flips, Gaussian blur, salt-and-pepper, non-uniform lighting.
- Known edge case: when a wire is colinear with a component edge, detectors may elongate the box. The generator can be constrained to 0/90/180/270° to increase robustness.

## YOLO (v8?)

    Standard backbone + FPN/PAN, anchor-free head.

    Trained ~20 epochs (640 px). Monitored: box_loss, cls_loss, dfl_loss, precision/recall, mAP@0.5, mAP@0.5:0.95.

## Faster R-CNN (ResNet-50 + RPN)

    Trained on generated data and annotated full schematics (COCO).

    Typical loss trends observed: train ↓ steadily; val ↓ to a stable plateau.


Training/validation losses and mAP (placeholder).


______________________________________________

Results (summary)

    Connectivity module accuracy: ~92% average; ~84% on dense scenes (≥6 components).

    Per-class accuracy (example, YOLO):

        C 99.4% · R 98.6% · Z 95.7% · L 91.2% · T 85.8% · D 84.1% · background 94.9%

    Qualitative samples and confusion matrix below.


Per-class confusion (placeholder).


Example NxN connection matrix (placeholder).
Connectivity pipeline (details)

    Read detector outputs (boxes, classes).

    Normalize brightness; equalize histogram.

    Threshold → morphology (dilation) → Canny edges.

    Extract contours, approximate by polylines; intersect with component boxes.

    Build NxN adjacency/connection matrix; apply simple heuristics (min wire length, angle sanity).

How to run

# 1) Train YOLOv8 (example)
yolo train data=ecad.yaml model=yolov8m.pt imgsz=640 epochs=20 project=runs/ecad

# 2) Inference
yolo predict model=runs/ecad/weights/best.pt source=dataset/test

# 3) Build connection matrix from detections
python tools/build_connections.py \
  --detections runs/ecad/preds.json \
  --images dataset/test/images \
  --out results/connections.json

Dataset layout

dataset/
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt      # YOLO format

Repository structure

ECAD_understanding/
├─ generator/                 # synthetic data generator
├─ training/
│  ├─ yolo/                   # YOLOv8 configs/scripts
│  └─ frcnn/                  # Faster R-CNN training
├─ connectivity/              # classical CV connectivity module
├─ tools/                     # build_connections.py, viz helpers
├─ dataset/                   # (optional) samples / pointers
├─ docs/
│  └─ img/                    # put screenshots here
└─ README.md

My role

    Co-designed the pipeline and experiments.

    Built the data generator (masks, affine/perspective, realistic wires).

    Trained/evaluated YOLOv8 / Faster R-CNN.

    Implemented the connectivity module in OpenCV and its metrics (accuracy of the connection matrix).

Limitations & next steps

    Box elongation for wire-colinear components (mitigate in generator and with data balancing).

    YOLO false positives on textured backgrounds → improve augmentations and class balance.

    Connectivity drops on very dense scenes → consider lightweight graph validation post-processing.

    Future: small GNN over detected symbols/anchors; better wire rendering; exporter to netlists.
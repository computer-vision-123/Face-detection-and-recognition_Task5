"""
pipeline/workers.py
-------------------
QObject workers that run C++ backend calls on a QThread,
keeping the UI responsive during heavy computation.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

try:
    import cv_backend
    BACKEND_OK = True
except ImportError:
    BACKEND_OK = False

# Resolved once at import time; used by RecognitionWorker.
def _find_model() -> str:
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:6]:
        for match in parent.rglob("pca_model.bin"):
            return str(match)
    return "pca_model.bin"

MODEL_PATH = _find_model()


class DetectionWorker(QObject):
    """Runs cv_backend.detect_faces off the main thread."""

    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, img, scale: float, neighbors: int) -> None:
        super().__init__()
        self.img       = img
        self.scale     = scale
        self.neighbors = neighbors

    def run(self) -> None:
        try:
            result = cv_backend.detect_faces(self.img, self.scale, self.neighbors)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class RecognitionWorker(QObject):
    """
    Loads the PCA model, builds a gallery from dataset_dir,
    projects the probe image, finds the nearest neighbour,
    and computes the ROC curve — all off the main thread.
    """

    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    _EXTS = ("*.pgm", "*.jpg", "*.jpeg", "*.png", "*.bmp")

    def __init__(self, dataset_dir: str, probe_img, n_components: int) -> None:
        super().__init__()
        self.dataset_dir  = dataset_dir
        self.probe_img    = probe_img
        self.n_components = n_components

    def run(self) -> None:
        try:
            model = cv_backend.PCAModel()
            model.load(MODEL_PATH)

            # ── build gallery ────────────────────────────────────────
            gallery_paths, gallery_labels = [], []
            label_id = 0
            dataset  = Path(self.dataset_dir)

            for subdir in sorted(dataset.iterdir()):
                if not subdir.is_dir():
                    continue
                paths = []
                for ext in self._EXTS:
                    paths.extend(sorted(subdir.glob(ext)))
                for img_path in paths:
                    gallery_paths.append(img_path)
                    gallery_labels.append(label_id)
                if paths:
                    label_id += 1

            if not gallery_paths:
                self.error.emit(
                    "No images (pgm/jpg/png) found in the selected dataset folder.")
                return

            # ── infer image dimensions from model mean ───────────────
            model_mean = np.asarray(model.get_mean())
            D = model_mean.shape[0]
            side_h = side_w = int(round(D ** 0.5))
            if side_h * side_w != D:
                for h in range(112, 32, -1):
                    if D % h == 0:
                        side_h, side_w = h, D // h
                        break

            # ── project gallery ──────────────────────────────────────
            gallery_imgs = []
            for p in gallery_paths:
                g = cv_backend.imread_gray(str(p))
                g = cv_backend.imresize(g, side_h, side_w)
                gallery_imgs.append(g.astype(np.float64).ravel())

            gallery_matrix = np.array(gallery_imgs, dtype=np.float64)
            gallery_labels = np.array(gallery_labels, dtype=np.int32)
            gallery_proj   = model.transform(gallery_matrix)

            # ── project probe ────────────────────────────────────────
            probe_gray = cv_backend.rgb_to_gray(self.probe_img)
            probe_gray = cv_backend.imresize(probe_gray, side_h, side_w)
            probe_vec  = probe_gray.astype(np.float64).ravel()[np.newaxis, :]
            probe_proj = model.transform(probe_vec)

            # ── 1-NN match ───────────────────────────────────────────
            dists      = np.linalg.norm(gallery_proj - probe_proj, axis=1)
            best_idx   = int(np.argmin(dists))
            best_dist  = float(dists[best_idx])
            best_label = int(gallery_labels[best_idx])

            subject_dirs = sorted(
                d.name for d in dataset.iterdir() if d.is_dir())
            subject_name = (subject_dirs[best_label]
                            if best_label < len(subject_dirs)
                            else str(best_label))

            # ── ROC ──────────────────────────────────────────────────
            roc = cv_backend.compute_roc(
                gallery_proj, gallery_proj,
                gallery_labels, gallery_labels,
            )

            self.finished.emit({
                "match_label":   best_label,
                "match_subject": subject_name,
                "match_dist":    best_dist,
                "roc":           roc,
            })
        except Exception:
            self.error.emit(traceback.format_exc())

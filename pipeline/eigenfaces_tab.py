"""
pipeline/eigenfaces_tab.py
---------------------------
QWidget for the Eigenfaces gallery tab.
Loads a PCA model and renders the first N eigenfaces in a scrollable grid.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QScrollArea,
    QGridLayout, QMessageBox,
)

from pipeline.ui_utils import numpy_to_pixmap
from pipeline.workers  import MODEL_PATH, BACKEND_OK

try:
    import cv_backend
except ImportError:
    pass

_MAX_FACES  = 64
_THUMB_SIZE = 80
_COLS       = 8


class EigenfacesTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._grid_layout = None
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(12)
        load_btn = QPushButton("Load Eigenfaces")
        load_btn.setMinimumHeight(36)
        load_btn.clicked.connect(self._load)
        top.addWidget(load_btn)
        top.addStretch()

        self._info_lbl = QLabel("")
        self._info_lbl.setStyleSheet(
            "color: #1E9BDB; font-size: 12px; font-weight: 600; "
            "background: #DFF0FA; border-radius: 6px; padding: 4px 10px;")
        top.addWidget(self._info_lbl)
        layout.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: 1.5px solid #B8D9F0; border-radius: 10px; background: #FFFFFF; }")
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: #FFFFFF;")
        scroll.setWidget(self._grid_widget)
        layout.addWidget(scroll)

    def _load(self) -> None:
        if not BACKEND_OK:
            QMessageBox.warning(self, "Backend Unavailable", "cv_backend not loaded.")
            return
        try:
            model = cv_backend.PCAModel()
            model.load(MODEL_PATH)
            ef   = np.asarray(model.get_eigenfaces())   # (D, k)
            k    = ef.shape[1]
            side = int(np.sqrt(ef.shape[0]))

            self._info_lbl.setText(f"{k} eigenfaces  |  {side}×{side} px")

            # Replace the grid layout
            if self._grid_layout:
                QWidget().setLayout(self._grid_layout)
            grid = QGridLayout(self._grid_widget)
            grid.setSpacing(6)
            self._grid_layout = grid

            for i in range(min(k, _MAX_FACES)):
                face = ef[:, i].reshape(side, side)
                mn, mx = face.min(), face.max()
                face = ((face - mn) / (mx - mn) * 255).astype(np.uint8) \
                       if mx > mn else np.zeros((side, side), dtype=np.uint8)
                face_rgb = np.stack([face] * 3, axis=-1)
                pix = numpy_to_pixmap(face_rgb).scaled(
                    _THUMB_SIZE, _THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl = QLabel()
                lbl.setPixmap(pix)
                lbl.setToolTip(f"Eigenface {i + 1}")
                lbl.setStyleSheet(
                    "border: 1px solid #B8D9F0; border-radius: 6px; "
                    "padding: 2px; background: #F5FBFF;")
                grid.addWidget(lbl, i // _COLS, i % _COLS)

        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

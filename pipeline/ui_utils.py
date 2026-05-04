"""
pipeline/ui_utils.py
--------------------
Shared UI helpers used across all tab widgets.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy


def numpy_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert (H,W,3) or (H,W) uint8 numpy array → QPixmap."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w, c = arr.shape
    img = QImage(arr.data, w, h, w * c, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def image_label(placeholder: str = "No image loaded") -> QLabel:
    """Create a styled canvas QLabel for displaying images."""
    lbl = QLabel(placeholder)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setMinimumSize(320, 240)
    lbl.setStyleSheet(
        "background: #FFFFFF;"
        "border: 1.5px solid #B8D9F0;"
        "border-radius: 10px;"
        "color: #7DC4E8;"
        "font-size: 13px;"
        "font-weight: 500;"
    )
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return lbl

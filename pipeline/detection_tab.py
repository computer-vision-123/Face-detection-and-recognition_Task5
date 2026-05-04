"""
pipeline/detection_tab.py
--------------------------
QWidget for the Face Detection tab.
Owns its own UI, wires a DetectionWorker on a QThread.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
    QGroupBox, QProgressBar, QFileDialog, QMessageBox,
)

from pipeline.ui_utils import numpy_to_pixmap, image_label
from pipeline.workers  import DetectionWorker, BACKEND_OK

try:
    import cv_backend
except ImportError:
    pass


class DetectionTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._img:    object  = None
        self._thread: QThread = None
        self._build()

    # ----------------------------------------------------------------
    #  UI
    # ----------------------------------------------------------------
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ── left panel ───────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(12)

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self._load_image)
        left.addWidget(load_btn)

        params = QGroupBox("Detection Parameters")
        pf = QVBoxLayout(params)
        pf.addWidget(QLabel("Scale factor:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(1.05, 2.0)
        self._scale_spin.setSingleStep(0.05)
        self._scale_spin.setValue(1.3)
        pf.addWidget(self._scale_spin)
        pf.addWidget(QLabel("Min neighbors:"))
        self._neighbors_spin = QSpinBox()
        self._neighbors_spin.setRange(1, 20)
        self._neighbors_spin.setValue(5)
        pf.addWidget(self._neighbors_spin)
        left.addWidget(params)

        self._detect_btn = QPushButton("Detect Faces")
        self._detect_btn.setEnabled(False)
        self._detect_btn.clicked.connect(self._run_detection)
        left.addWidget(self._detect_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        left.addWidget(self._progress)

        self._result_lbl = QLabel("")
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setStyleSheet(
            "color: #005C99; font-size: 12px; font-weight: 600; "
            "background: #DFF0FA; border-radius: 7px; padding: 6px 10px;"
        )
        left.addWidget(self._result_lbl)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setFixedWidth(200)
        left_widget.setLayout(left)

        # ── right canvas ─────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)
        img_row = QHBoxLayout()
        img_row.setSpacing(12)

        col1 = QVBoxLayout()
        col1.addWidget(QLabel("Input"))
        self._input_lbl = image_label("Load an image to begin")
        col1.addWidget(self._input_lbl)

        col2 = QVBoxLayout()
        col2.addWidget(QLabel("Detection Result"))
        self._output_lbl = image_label("Result will appear here")
        col2.addWidget(self._output_lbl)

        img_row.addLayout(col1)
        img_row.addLayout(col2)
        right.addLayout(img_row)

        root.addWidget(left_widget)
        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget)

    # ----------------------------------------------------------------
    #  Slots
    # ----------------------------------------------------------------
    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.pgm)")
        if not path:
            return
        if BACKEND_OK:
            self._img = cv_backend.imread_rgb(path)
        else:
            from PIL import Image
            self._img = np.array(Image.open(path).convert("RGB"))
        self._input_lbl.setPixmap(
            numpy_to_pixmap(self._img).scaled(
                self._input_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._detect_btn.setEnabled(True)
        self._result_lbl.setText("")

    def _run_detection(self) -> None:
        if self._img is None or not BACKEND_OK:
            return
        self._detect_btn.setEnabled(False)
        self._progress.setVisible(True)

        self._thread = QThread()
        self._worker = DetectionWorker(
            self._img, self._scale_spin.value(), self._neighbors_spin.value())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_result(self, result) -> None:
        self._progress.setVisible(False)
        self._detect_btn.setEnabled(True)
        vis = np.asarray(result.visualisation)
        self._output_lbl.setPixmap(
            numpy_to_pixmap(vis).scaled(
                self._output_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._result_lbl.setText(
            f"Faces detected: {result.num_faces}\n"
            f"Computation: {result.computation_time_ms:.1f} ms")

    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._detect_btn.setEnabled(True)
        QMessageBox.critical(self, "Detection Error", msg)

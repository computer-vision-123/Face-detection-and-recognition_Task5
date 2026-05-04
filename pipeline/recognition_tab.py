"""
pipeline/recognition_tab.py
----------------------------
QWidget for the Face Recognition tab + ROCDialog popup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSpinBox,
    QGroupBox, QFormLayout, QProgressBar,
    QFileDialog, QMessageBox, QDialog, QFrame,
)

from pipeline.ui_utils import numpy_to_pixmap, image_label
from pipeline.workers  import RecognitionWorker, MODEL_PATH, BACKEND_OK

try:
    import cv_backend
except ImportError:
    pass


class ROCDialog(QDialog):
    """Modal dialog that renders the ROC curve using matplotlib."""

    def __init__(self, roc_result, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ROC Curve")
        self.setMinimumSize(560, 440)
        self.roc = roc_result
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(10)

        title = QLabel("Receiver Operating Characteristic")
        title.setFont(QFont("Segoe UI Semibold", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0A7EC2; font-weight: 700;")
        layout.addWidget(title)

        info = QLabel(
            f"AUC = {self.roc.auc:.4f}        "
            f"Genuine pairs: {self.roc.n_genuine}        "
            f"Impostor pairs: {self.roc.n_impostor}"
        )
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet(
            "color: #3A6080; font-size: 11px; margin-bottom: 6px; "
            "background: #DFF0FA; border-radius: 6px; padding: 5px;"
        )
        layout.addWidget(info)

        try:
            self._add_matplotlib(layout)
        except Exception:
            self._add_text_fallback(layout)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        h = QHBoxLayout()
        h.addStretch()
        h.addWidget(close_btn)
        layout.addLayout(h)

    def _add_matplotlib(self, layout) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        fpr = np.asarray(self.roc.fpr)
        tpr = np.asarray(self.roc.tpr)

        fig, ax = plt.subplots(figsize=(5, 3.6), dpi=110)
        fig.patch.set_facecolor("#F5FBFF")
        ax.set_facecolor("#F0F7FC")
        ax.plot(fpr, tpr, color="#0A7EC2", lw=2.5, label=f"AUC = {self.roc.auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.10, color="#1E9BDB")
        ax.plot([0, 1], [0, 1], "--", color="#7DC4E8", lw=1)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(True, alpha=0.25, color="#A8D5EF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#B8D9F0")
        ax.spines["bottom"].set_color("#B8D9F0")
        fig.tight_layout()
        layout.addWidget(FigureCanvasQTAgg(fig))

    def _add_text_fallback(self, layout) -> None:
        lbl = QLabel(f"matplotlib not available.\nAUC = {self.roc.auc:.4f}")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)


class RecognitionTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._probe_img:   object = None
        self._dataset_dir: str    = None
        self._roc_data:    object = None
        self._thread:      QThread = None
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

        # model status
        model_grp = QGroupBox("PCA Model")
        mf = QVBoxLayout(model_grp)
        model_exists = Path(MODEL_PATH).exists()
        path_lbl = QLabel(str(Path(MODEL_PATH)))
        path_lbl.setStyleSheet("color: #6A90AA; font-size: 10px; font-style: italic;")
        path_lbl.setWordWrap(True)
        mf.addWidget(path_lbl)
        status_lbl = QLabel(
            "Model Found" if model_exists
            else "NOT FOUND – place pca_model.bin next to main.py")
        status_lbl.setStyleSheet(
            "color: #4DB8FF; font-size: 11px; font-weight: 700;" if model_exists
            else "color: #F87171; font-size: 11px; font-weight: 700;")
        mf.addWidget(status_lbl)
        left.addWidget(model_grp)

        # dataset picker
        ds_grp = QGroupBox("Dataset")
        df = QVBoxLayout(ds_grp)
        self._ds_lbl = QLabel("No dataset selected")
        self._ds_lbl.setStyleSheet("color: #7DC4E8; font-size: 11px; font-style: italic;")
        self._ds_lbl.setWordWrap(True)
        df.addWidget(self._ds_lbl)
        ds_btn = QPushButton("Select Dataset")
        ds_btn.clicked.connect(self._select_dataset)
        df.addWidget(ds_btn)
        left.addWidget(ds_grp)

        probe_btn = QPushButton("Load Probe")
        probe_btn.clicked.connect(self._load_probe)
        left.addWidget(probe_btn)

        params = QGroupBox("Parameters")
        pf = QFormLayout(params)
        self._n_comp_spin = QSpinBox()
        self._n_comp_spin.setRange(5, 200)
        self._n_comp_spin.setValue(50)
        pf.addRow("Components:", self._n_comp_spin)
        left.addWidget(params)

        self._run_btn = QPushButton("Run Recognition")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_recognition)
        left.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        left.addWidget(self._progress)

        self._result_lbl = QLabel("")
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setStyleSheet(
            "color: #005C99; font-size: 12px; font-weight: 600; "
            "background: #DFF0FA; border-radius: 7px; padding: 6px 10px;")
        left.addWidget(self._result_lbl)

        self._roc_btn = QPushButton("View ROC Curve")
        self._roc_btn.setEnabled(False)
        self._roc_btn.clicked.connect(self._show_roc)
        left.addWidget(self._roc_btn)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setFixedWidth(230)
        left_widget.setLayout(left)

        # ── probe canvas ─────────────────────────────────────────────
        probe_frame = QFrame()
        probe_frame.setStyleSheet(
            "QFrame { background: #FFFFFF; border: 1.5px solid #B8D9F0; border-radius: 12px; }")
        probe_lay = QVBoxLayout(probe_frame)
        probe_lay.setContentsMargins(0, 0, 0, 0)
        probe_lay.setSpacing(0)
        probe_hdr = QLabel("  Probe Image")
        probe_hdr.setFixedHeight(34)
        probe_hdr.setStyleSheet(
            "background: #EEF5FB; color: #0A7EC2; font-size: 11px; font-weight: 700;"
            "letter-spacing: 0.08em; border-bottom: 1px solid #B8D9F0;"
            "border-radius: 0px; padding-left: 12px;")
        probe_lay.addWidget(probe_hdr)
        self._probe_lbl = image_label("Load a probe image")
        self._probe_lbl.setStyleSheet(
            "background: transparent; border: none; color: #7DC4E8; font-size: 13px; font-weight: 500;")
        self._probe_lbl.setContentsMargins(8, 8, 8, 8)
        probe_lay.addWidget(self._probe_lbl)

        root.addWidget(left_widget)
        root.addWidget(probe_frame, 1)

    # ----------------------------------------------------------------
    #  Slots
    # ----------------------------------------------------------------
    def _select_dataset(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if d:
            self._dataset_dir = d
            self._ds_lbl.setText(Path(d).name)
            self._ds_lbl.setStyleSheet("color: #005C99; font-size: 11px; font-weight: 600;")
            self._update_run_btn()

    def _load_probe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Probe Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.pgm)")
        if not path:
            return
        if BACKEND_OK:
            self._probe_img = cv_backend.imread_rgb(path)
        else:
            from PIL import Image
            import numpy as np
            self._probe_img = np.array(Image.open(path).convert("RGB"))
        self._probe_lbl.setPixmap(
            numpy_to_pixmap(self._probe_img).scaled(
                self._probe_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._update_run_btn()

    def _update_run_btn(self) -> None:
        self._run_btn.setEnabled(
            self._probe_img is not None
            and self._dataset_dir is not None
            and BACKEND_OK)

    def _run_recognition(self) -> None:
        self._run_btn.setEnabled(False)
        self._roc_btn.setEnabled(False)
        self._progress.setVisible(True)

        self._thread = QThread()
        self._worker = RecognitionWorker(
            self._dataset_dir, self._probe_img, self._n_comp_spin.value())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_result(self, data: dict) -> None:
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._roc_data = data["roc"]
        self._roc_btn.setEnabled(True)
        subject = data.get("match_subject", str(data["match_label"]))
        dist    = data["match_dist"]
        self._result_lbl.setText(f"Best match: {subject}\nDistance:   {dist:.4f}")

    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Recognition Error", msg)

    def _show_roc(self) -> None:
        if self._roc_data is not None:
            ROCDialog(self._roc_data, parent=self).exec_()

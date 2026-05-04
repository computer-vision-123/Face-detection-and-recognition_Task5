"""
pipeline/app_orchestrator.py
-----------------------------
Thin top-level orchestrator.
Builds the main window shell (header + tabs + status bar)
and delegates all tab logic to the dedicated tab modules.
"""

from __future__ import annotations

from PyQt5.QtCore    import Qt
from PyQt5.QtGui     import QFont
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel,
    QMainWindow, QStatusBar, QTabWidget,
    QVBoxLayout, QWidget,
)

from pipeline.workers          import BACKEND_OK
from pipeline.detection_tab    import DetectionTab
from pipeline.recognition_tab  import RecognitionTab
from pipeline.eigenfaces_tab   import EigenfacesTab


class AppOrchestrator:
    """
    Wires the QMainWindow shell:
      • gradient header with backend status badge
      • tab widget (Detection / Recognition / Eigenfaces)
      • status bar
    All tab behaviour lives in the respective tab modules.
    """

    def __init__(self, window: QMainWindow) -> None:
        self._win = window
        window.setWindowTitle("FaceScope")
        window.setMinimumSize(960, 660)
        window.menuBar().setVisible(False)

        central = QWidget()
        window.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_tabs())
        window.setStatusBar(self._build_statusbar())

    # ----------------------------------------------------------------
    #  Private builders
    # ----------------------------------------------------------------
    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setFixedHeight(80)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #012840, stop:0.5 #013558, stop:1 #012840);
                border-bottom: 2px solid #1E9BDB;
            }
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)

        # backend badge
        badge_bg  = "rgba(52,211,153,0.18)" if BACKEND_OK else "rgba(239,68,68,0.18)"
        badge_col = "#4DB8FF"               if BACKEND_OK else "#F87171"
        badge_txt = "Engine Ready"          if BACKEND_OK else "Engine Offline"
        badge = QLabel(badge_txt)
        badge.setStyleSheet(
            f"color: {badge_col}; background: {badge_bg}; "
            f"border: 1px solid {badge_col}; border-radius: 12px; "
            f"font-size: 11px; font-weight: 600; padding: 4px 12px;")
        badge.setFixedWidth(130)
        layout.addWidget(badge)
        layout.addStretch(1)

        # title
        title = QLabel("FaceScope")
        title.setFont(QFont("Segoe UI", 48, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #DEEFFE; background: transparent; "
            "letter-spacing: 0.06em; font-weight: 700;")
        layout.addWidget(title)
        layout.addStretch(1)

        # right spacer (keeps title centred)
        spacer = QLabel("")
        spacer.setFixedWidth(130)
        layout.addWidget(spacer)

        return header

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: #EEF5FB; }
            QTabBar { background: #FFFFFF; border-bottom: 1px solid #B8D9F0; }
            QTabBar::tab {
                background: transparent;
                color: #6A90AA;
                padding: 14px 34px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.04em;
                border: none;
                border-bottom: 3px solid transparent;
                min-width: 130px;
            }
            QTabBar::tab:selected {
                color: #0A7EC2;
                border-bottom: 3px solid #0A7EC2;
                font-weight: 700;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(10,126,194,0.08), stop:1 transparent);
            }
            QTabBar::tab:hover:!selected {
                color: #1E9BDB;
                border-bottom: 3px solid rgba(30,155,219,0.35);
                background: rgba(30,155,219,0.04);
            }
        """)
        tabs.addTab(DetectionTab(),   "Detection")
        tabs.addTab(RecognitionTab(), "Recognition")
        tabs.addTab(EigenfacesTab(),  "Eigenfaces")
        self._tabs = tabs
        return tabs

    def _build_statusbar(self) -> QStatusBar:
        bar = QStatusBar()
        bar.setStyleSheet(
            "font-size: 11px; color: #6A90AA; "
            "background: #FFFFFF; border-top: 1px solid #B8D9F0; padding-left: 6px;")
        bar.showMessage("Ready    FaceScope v1.0")
        return bar

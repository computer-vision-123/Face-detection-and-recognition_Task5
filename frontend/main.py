import sys
from pathlib import Path

# ── resolve project root so 'pipeline' and 'backend' are importable ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# If cv_backend.pyd/.so lives in frontend/ (CMake output), also add that:
sys.path.insert(0, str(ROOT / "frontend"))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow

from pipeline.app_orchestrator import AppOrchestrator


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)

    # Load stylesheet
    qss_path = Path(__file__).parent / "ui" / "theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text())

    window = QMainWindow()
    AppOrchestrator(window)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

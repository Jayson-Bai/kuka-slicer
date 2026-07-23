"""Qt application entry point."""

from __future__ import annotations

import sys

from python_qt_binding import QtWidgets

from .ui import ExternalNpzPreprocessorWindow


def main(argv=None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or sys.argv)
    window = ExternalNpzPreprocessorWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from main_window import MainWindow
from src.utils import str2bool
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--use-server", type=str2bool,  nargs='?', const=True, required=False, help="use http server", default=False)
ap.add_argument("-ip", "--ip", required=False, help="http server ip")
ap.add_argument("-p", "--port", required=False, help="http server port")

if __name__ == '__main__':
    app = QApplication([])

    main_widget = MainWindow(args=ap.parse_args())
    main_widget.show()

    # enable dark mode
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)

    app.setPalette(palette)

    app.exec()

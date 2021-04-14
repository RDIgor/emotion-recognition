from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QMenuBar,
    QMenu,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog)

from PyQt5.QtWidgets import QAction
from markup_widget import MarkupWidget
from markup_loader_widget import MarkupLoaderWidget


DEFAULT_WINDOW_WIDTH = 1700
DEFAULT_WINDOW_HEIGHT = 900


class MainWindow(QMainWindow):
    def __init__(self, parent=None, args=None):
        super(QWidget, self).__init__(parent)
        self.central_widget = MarkupWidget(self, args)
        self.markup_loader_widget = MarkupLoaderWidget(self)

        # create menubar
        self.create_menubar()

        # set central widget
        self.setCentralWidget(self.central_widget)

        self.setMinimumSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setWindowTitle('Markup tool')

    def create_menubar(self):
        menubar = QMenuBar(self)

        # create actions
        open_action = QAction('&Open...', self)
        exit_action = QAction('&Exit', self)
        load_markup_action = QAction('&Load', self)

        # connections
        open_action.triggered.connect(self.open_file)
        exit_action.triggered.connect(self.exit)
        load_markup_action.triggered.connect(self.load_markup)

        # create menu
        file_menu = QMenu('&File', self)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        markup_menu = QMenu('&Markup', self)
        markup_menu.addAction(load_markup_action)

        # add menu
        menubar.addMenu(file_menu)
        menubar.addMenu(markup_menu)

        # set menubar
        self.setMenuBar(menubar)

    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home/', 'Source file (*.mov *.mp4 *.jpg *.jpeg '
                                                                         '*.png')
        filepath = fname[0]

        if len(filepath) > 0:
            self.central_widget.set_source(filepath)

    def load_markup(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select folder')

        if folder_path is not None:
            self.markup_loader_widget.set_markup_folder(folder_path)
            self.setCentralWidget(self.markup_loader_widget)

    def exit(self):
        self.close()

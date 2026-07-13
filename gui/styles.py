"""Shared visual constants for the PyQt6 interface."""

APP_STYLESHEET = """
QWidget {
    background: #10131a;
    color: #eef1f6;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
}
QMainWindow, QFrame#appRoot {
    background: #0c0f15;
}
QFrame#header, QFrame#panel, QFrame#toolbar {
    background: #151a23;
    border: 1px solid #242b38;
    border-radius: 10px;
}
QLabel#appTitle {
    font-size: 18pt;
    font-weight: 700;
}
QLabel#sectionTitle {
    font-size: 12pt;
    font-weight: 650;
}
QLabel#muted, QLabel#statusValue {
    color: #aab3c2;
}
QLabel#statusValue {
    background: #0e1219;
    border: 1px solid #2a3241;
    border-radius: 8px;
    padding: 5px 9px;
}
QPushButton, QToolButton {
    background: #26334a;
    border: 1px solid #34445f;
    border-radius: 7px;
    min-height: 30px;
    padding: 3px 12px;
    font-weight: 600;
}
QPushButton:hover, QToolButton:hover {
    background: #31415e;
}
QPushButton:pressed, QToolButton:pressed {
    background: #1c2739;
}
QPushButton:disabled, QToolButton:disabled {
    color: #687284;
    background: #1a202b;
    border-color: #242b38;
}
QPushButton#primaryButton {
    background: #3266d5;
    border-color: #4779df;
}
QPushButton#primaryButton:hover {
    background: #3e73df;
}
QLineEdit {
    background: #0d1118;
    border: 1px solid #30394a;
    border-radius: 7px;
    min-height: 30px;
    padding: 2px 9px;
    selection-background-color: #3266d5;
}
QLineEdit:focus {
    border-color: #5a86e4;
}
QListWidget, QTableView {
    background: #0d1118;
    alternate-background-color: #111722;
    border: 1px solid #252d3a;
    border-radius: 8px;
    outline: none;
}
QListWidget::item {
    min-height: 30px;
    padding: 3px 8px;
    border-radius: 5px;
}
QListWidget::item:selected {
    background: #2f5ebc;
    color: white;
}
QHeaderView::section {
    background: #1a202b;
    color: #dce2ec;
    border: 0;
    border-right: 1px solid #2d3543;
    border-bottom: 1px solid #2d3543;
    padding: 7px;
    font-weight: 600;
}
QTableView {
    gridline-color: #2a3240;
    selection-background-color: #315ea9;
    selection-color: white;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #111620;
    border: 0;
    margin: 0;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #3a4456;
    border-radius: 5px;
    min-height: 26px;
    min-width: 26px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}
QSplitter::handle {
    background: #222a36;
    width: 5px;
    height: 5px;
}
QStatusBar {
    background: #0c0f15;
    color: #8f99aa;
}
QToolTip {
    background: #202735;
    color: #f4f6fa;
    border: 1px solid #3a465b;
    padding: 5px;
}
"""

APP_STYLESHEET += """
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: #0d1118;
    border: 1px solid #30394a;
    border-radius: 7px;
    min-height: 30px;
    padding: 2px 8px;
    selection-background-color: #3266d5;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border-color: #5a86e4;
}
QComboBox::drop-down {
    border: 0;
    width: 24px;
}
QComboBox QAbstractItemView {
    background: #111722;
    border: 1px solid #30394a;
    selection-background-color: #315ea9;
}
QTabWidget::pane {
    background: #10151e;
    border: 1px solid #293242;
    border-radius: 8px;
    top: -1px;
}
QTabBar::tab {
    background: #171d28;
    color: #aeb8c8;
    border: 1px solid #293242;
    border-bottom: 0;
    padding: 8px 13px;
    margin-right: 2px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}
QTabBar::tab:selected {
    background: #26334a;
    color: #ffffff;
}
QTabBar::tab:hover:!selected {
    background: #202837;
}
QGroupBox {
    background: #121822;
    border: 1px solid #293242;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #dfe5ef;
}
QCheckBox {
    spacing: 7px;
}
QCheckBox::indicator {
    width: 17px;
    height: 17px;
}
QScrollArea {
    background: transparent;
    border: 0;
}
QPlainTextEdit#bufferCurrent {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 9.5pt;
}
QLabel[status="ok"] {
    color: #77d69b;
}
QLabel[status="warning"] {
    color: #e7bd67;
}
QLabel[status="error"] {
    color: #f08080;
}
QFrame#separator {
    color: #343d4d;
    max-width: 1px;
}
"""

APP_STYLESHEET += """
QSlider::groove:horizontal {
    border: 1px solid #30394a;
    height: 6px;
    background: #111722;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #3266d5;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e2e8f0;
    border: 1px solid #64748b;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
"""

APP_STYLESHEET += """
QFrame#bottomNavigation {
    background: #111722;
    border: 1px solid #293242;
    border-radius: 12px;
}
QPushButton#navigationButton {
    background: #1b2433;
    color: #93a4ba;
    border: 1px solid #334155;
    border-radius: 10px;
    font-size: 11pt;
    font-weight: 700;
    padding: 7px 22px;
}
QPushButton#navigationButton:checked {
    background: #2563eb;
    color: white;
    border-color: #60a5fa;
}
QPushButton#navigationButton:hover:!checked {
    background: #26354a;
    color: #e2e8f0;
}
QFrame#controlGroup {
    background: #101722;
    border: 1px solid #303b4d;
    border-radius: 9px;
}
QLabel#groupTitle {
    color: #e2e8f0;
    font-weight: 700;
    padding-right: 4px;
}
QPushButton#exportButton {
    background: #172033;
    border: 1px solid #3b4b67;
    min-height: 36px;
}
QFrame#modeToggle {
    background: #0b1018;
    border: 1px solid #334155;
    border-radius: 8px;
}
QPushButton#modeSegment {
    background: transparent;
    color: #64748b;
    border: 0;
    border-radius: 6px;
    min-height: 28px;
    min-width: 76px;
    padding: 2px 10px;
}
QPushButton#modeSegment:checked {
    background: #2563eb;
    color: #ffffff;
}
QPushButton#modeSegment:hover:!checked {
    background: #1e293b;
    color: #cbd5e1;
}
QTreeWidget {
    background: #0d1118;
    border: 1px solid #252d3a;
    border-radius: 8px;
    outline: none;
}
QTreeWidget::item {
    min-height: 30px;
    padding: 3px 6px;
    border-radius: 5px;
}
QTreeWidget::item:selected {
    background: #2f5ebc;
    color: white;
}
"""

APP_STYLESHEET += """
QLabel#appTitle, QLabel#sectionTitle, QLabel#groupTitle {
    background: transparent;
    border: none;
    padding: 0px;
}
"""

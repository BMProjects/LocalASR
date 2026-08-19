"""Shared visual language and small presentation helpers for the desktop apps.

One dark theme, applied identically to all three entry points.

Two things are easy to get wrong in a Qt stylesheet and both were wrong before:

* A combo box's popup is a separate `QAbstractItemView`, not covered by `QComboBox`
  rules. Styling only the closed box leaves the open list painted by the platform
  palette — dark text on a dark background, which is what made it unreadable.
* Qt draws check-box and radio indicators from the palette, not the stylesheet, unless
  the indicator itself is styled. On a dark background an unstyled indicator is a dark
  square on dark, invisible until clicked.

So the palette is set alongside the stylesheet: anything a rule misses still lands on
dark-appropriate colours instead of the light defaults.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

# --- palette ----------------------------------------------------------------
BG = "#16191f"
SURFACE = "#1e232b"
SURFACE_HIGH = "#262c36"
BORDER = "#333b47"
BORDER_STRONG = "#465162"
TEXT = "#e4e8ef"
TEXT_MUTED = "#9aa5b5"
TEXT_STRONG = "#f4f7fb"
ACCENT = "#4c9aff"
ACCENT_HOVER = "#69adff"
ACCENT_TEXT = "#0b1520"
STOP = "#ff8080"
STOP_TEXT = "#2a0d0f"
"""Bright fill with dark text, matching the accent button, so the two read as the same
kind of control differing only in what they do."""

APP_STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-size: 14px;
}}
QWidget#appSurface {{
    background: {BG};
}}
QLabel#pageTitle {{
    color: {TEXT_STRONG};
    font-size: 24px;
    font-weight: 700;
}}
QLabel#pageSubtitle, QLabel[muted="true"] {{
    color: {TEXT_MUTED};
}}
QFrame#card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#dropZone {{
    background: {SURFACE};
    border: 2px dashed {BORDER_STRONG};
    border-radius: 12px;
}}
QLabel#sectionTitle {{
    color: {TEXT_STRONG};
    font-size: 15px;
    font-weight: 650;
}}
QLabel#statusPill {{
    border-radius: 9px;
    padding: 4px 10px;
    font-weight: 650;
}}
QLabel#statusPill[tone="neutral"] {{ background: #2b323d; color: #c2cbd8; }}
QLabel#statusPill[tone="success"] {{ background: #16402f; color: #6ee7b0; }}
QLabel#statusPill[tone="active"]  {{ background: #16334f; color: #7cc0ff; }}
QLabel#statusPill[tone="warning"] {{ background: #4a3a15; color: #ffd479; }}
QLabel#statusPill[tone="danger"]  {{ background: #4c1f22; color: #ff9a9a; }}

QPushButton {{
    min-height: 34px;
    padding: 0 14px;
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    background: {SURFACE_HIGH};
}}
QPushButton:hover {{ background: #303845; border-color: #5a6779; }}
QPushButton:pressed {{ background: #222831; }}
QPushButton:disabled {{ color: #6b7484; background: #20252d; border-color: #2c333d; }}
QPushButton[role="primary"] {{
    color: {ACCENT_TEXT};
    background: {ACCENT};
    border-color: {ACCENT};
    font-weight: 650;
}}
QPushButton[role="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[role="primary"]:pressed {{ background: #3d84e0; }}
QPushButton[role="danger"] {{ color: #ff9a9a; border-color: #6a3438; }}
/* Filled, unlike `danger`, because this is the primary action rather than a secondary
   escape: when the same button toggles between starting and stopping, colour is what
   says which one it is about to do. */
QPushButton[role="stop"] {{
    color: {STOP_TEXT};
    background: {STOP};
    border-color: {STOP};
    font-weight: 650;
}}
QPushButton[role="stop"]:hover {{ background: #ff9494; border-color: #ff9494; }}
QPushButton[role="stop"]:pressed {{ background: #e56a6a; }}
QPushButton[role="primary"]:disabled,
QPushButton[role="danger"]:disabled,
QPushButton[role="stop"]:disabled {{
    color: #6b7484; background: #20252d; border-color: #2c333d;
}}

QListWidget, QTextEdit, QComboBox, QLineEdit, QAbstractSpinBox {{
    color: {TEXT};
    background: {SURFACE_HIGH};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
}}
QListWidget {{ padding: 5px; }}
QListWidget::item {{ padding: 8px; border-radius: 6px; }}
QListWidget::item:selected {{ background: #24425f; color: {TEXT_STRONG}; }}
QTextEdit {{ padding: 10px; }}
QTextEdit:disabled, QListWidget:disabled, QComboBox:disabled {{
    color: #6b7484; background: #1b1f26; border-color: #2c333d;
}}

QComboBox {{ min-height: 32px; padding: 0 9px; }}
QComboBox:hover {{ border-color: #5a6779; }}
/* The drop-down subcontrol is left entirely to Fusion, which draws the arrow from the
   palette. Declaring the subcontrol at all suppresses that arrow, and rebuilding it out
   of CSS borders renders as a stray dash at this size. */
/* The popup is a separate view; without these rules it keeps the platform palette and
   renders dark text on a dark background. */
QComboBox QAbstractItemView {{
    color: {TEXT};
    background: {SURFACE_HIGH};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
}}
QComboBox QAbstractItemView::item {{ min-height: 26px; padding: 4px 8px; }}
QComboBox QAbstractItemView::item:hover {{ background: #33415a; color: {TEXT_STRONG}; }}

/* Indicators come from the palette unless styled, and a dark square on dark is
   indistinguishable from no check box at all. */
QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER_STRONG};
    background: {SURFACE_HIGH};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: #20252d; border-color: #2c333d;
}}

QProgressBar {{
    min-height: 8px;
    max-height: 8px;
    border: 0;
    border-radius: 4px;
    background: #2b323d;
    text-align: center;
}}
QProgressBar::chunk {{ border-radius: 4px; background: {ACCENT}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #3c4553; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #4d596b; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #3c4553; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* Menus and dialogs are top-level windows and do not inherit #appSurface. */
QMenu {{
    color: {TEXT};
    background: {SURFACE_HIGH};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT}; color: {ACCENT_TEXT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}
QDialog, QMessageBox {{ background: {SURFACE}; color: {TEXT}; }}
QMessageBox QLabel {{ color: {TEXT}; }}
QToolTip {{
    color: {TEXT};
    background: #0f1319;
    border: 1px solid {BORDER_STRONG};
    padding: 5px;
}}
"""


def _dark_palette() -> QPalette:
    """Backstop for anything the stylesheet does not reach.

    Native pieces — file dialogs, some item views — read the palette directly, and a
    light default there shows up as white panels inside a dark window.
    """
    palette = QPalette()
    window, base, text = QColor(BG), QColor(SURFACE_HIGH), QColor(TEXT)
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_HIGH))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0f1319"))
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(ACCENT_TEXT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
    ):
        palette.setColor(disabled, role, QColor("#6b7484"))
    return palette


def apply_theme(app: QApplication) -> None:
    """Apply one predictable dark theme to all three entry points."""
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(APP_STYLESHEET)


def set_tone(widget: QWidget, tone: str) -> None:
    """Change a dynamic Qt style property and repaint it immediately."""
    widget.setProperty("tone", tone)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_role(widget: QWidget, role: str) -> None:
    """Set a button's role and repaint it.

    Qt does not re-evaluate a stylesheet when a dynamic property changes, so a role
    switched at runtime — start becoming stop — would keep the old colour without the
    unpolish/polish pair.
    """
    widget.setProperty("role", role)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()

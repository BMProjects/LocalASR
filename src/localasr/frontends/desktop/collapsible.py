"""A section that can fold away, with its state still readable when folded.

Setup — which models are loaded, which microphone — matters when you are arranging a
session and never again once it is arranged. It was taking two cards at the top of every
window, above the text the application exists to show.

Folding it away is only worth doing if the fold does not hide the answer to "is this
thing working". So the header carries a one-line summary: collapsed, it says what the
expanded panel would have said, in the space of a title.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QFrame):
    """One header row and a body that hides."""

    def __init__(self, title: str, *, expanded: bool = False) -> None:
        super().__init__()
        self.setObjectName("card")

        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggle.setAutoRaise(True)
        self.toggle.clicked.connect(self._toggled)

        self.summary = QLabel()
        self.summary.setProperty("muted", True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Elide rather than grow: a long node URL must not widen the window.
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        header.addWidget(self.toggle)
        header.addWidget(self.summary, 1)
        # Holds the title at the left once the summary hides on expand; without it the
        # header centres itself and the section appears to move when opened.
        header.addStretch(0)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 6, 0, 0)
        self._body_layout.setSpacing(9)
        self._body.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self._body)

    def add(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def set_summary(self, text: str) -> None:
        self.summary.setText(text)
        self.summary.setToolTip(text)

    @property
    def expanded(self) -> bool:
        return self.toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)
        self._toggled()

    def _toggled(self) -> None:
        expanded = self.toggle.isChecked()
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._body.setVisible(expanded)
        # The summary would repeat what the open panel already shows.
        self.summary.setVisible(not expanded)

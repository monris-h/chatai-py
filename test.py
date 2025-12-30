import html
import os
import re
import sys
import time
from datetime import datetime

from groq import Groq
from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

SYSTEM_PROMPT = "You are Monris AI, a helpful assistant."

BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
AVAILABLE_MODELS = [
    "allam-2-7b",
    "canopylabs/orpheus-arabic-saudi",
    "canopylabs/orpheus-v1-english",
    "groq/compound",
    "groq/compound-mini",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3-32b",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
]
EXCLUDED_MODELS = {
    "moonshotai/kimi-k2-instruct",
    "playai-tts",
    "playai-tts-arabic",
}
MODEL_OPTIONS = [model for model in AVAILABLE_MODELS if model not in EXCLUDED_MODELS]
DEFAULT_TEMPERATURE = 0.4
COMPACT_MAX_MESSAGES = 10
PROFILE_PRESETS = {
    "QA": {
        "prompt": (
            "You are Monris AI, a QA specialist. Focus on test plans, edge cases, automation, "
            "and clear acceptance criteria."
        ),
        "temperature": 0.2,
    },
    "Dev": {
        "prompt": (
            "You are Monris AI, a senior software engineer. Provide pragmatic guidance, "
            "clean code suggestions, and trade-offs."
        ),
        "temperature": 0.6,
    },
    "Support": {
        "prompt": (
            "You are Monris AI, a customer support assistant. Be concise, friendly, and "
            "solution-oriented."
        ),
        "temperature": 0.4,
    },
}
PRICING_USD_PER_MILLION = {}


def _format_inline(text: str) -> str:
    escaped = html.escape(text)
    return BOLD_PATTERN.sub(r"<b>\1</b>", escaped)


def format_assistant_message(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    parts: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith(("* ", "- ")):
            if not in_list:
                parts.append('<ul style="margin: 6px 0 6px 16px; padding: 0;">')
                in_list = True
            item_text = stripped[2:].strip()
            parts.append(f"<li>{_format_inline(item_text)}</li>")
            continue

        close_list()

        if stripped.startswith("### "):
            title = stripped[4:].strip()
            parts.append(
                f'<span style="font-size: 13px; font-weight: 700;">{_format_inline(title)}</span><br>'
            )
        elif stripped.startswith("## "):
            title = stripped[3:].strip()
            parts.append(
                f'<span style="font-size: 14px; font-weight: 700;">{_format_inline(title)}</span><br>'
            )
        elif stripped.startswith("# "):
            title = stripped[2:].strip()
            parts.append(
                f'<span style="font-size: 16px; font-weight: 800;">{_format_inline(title)}</span><br>'
            )
        elif stripped == "":
            parts.append("<br>")
        else:
            parts.append(f"{_format_inline(line)}<br>")

    close_list()
    return "".join(parts)


def usage_to_dict(usage) -> dict[str, int] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def estimate_cost(model: str, usage: dict[str, int] | None) -> float | None:
    if not usage:
        return None
    rates = PRICING_USD_PER_MILLION.get(model)
    if not rates:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    input_rate, output_rate = rates
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


def format_meta_text(meta: dict | None) -> str:
    if not meta:
        return ""
    parts = []
    latency = meta.get("latency")
    if latency is not None:
        parts.append(f"Latency {latency:.2f}s")
    usage = meta.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is not None and completion_tokens is not None:
        parts.append(f"Tokens {prompt_tokens}/{completion_tokens}")
    elif total_tokens is not None:
        parts.append(f"Tokens {total_tokens}")
    else:
        parts.append("Tokens N/A")
    cost = meta.get("cost")
    if cost is not None:
        parts.append(f"Cost ${cost:.6f}")
    else:
        parts.append("Cost N/A")
    return " | ".join(parts)


def build_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY before running this script.")
    return Groq(api_key=api_key)


class ChatWorker(QObject):
    response_delta = Signal(int, str)
    response_ready = Signal(int, str, dict)
    error = Signal(int, str)
    done = Signal(int)

    def __init__(
        self,
        client: Groq,
        session_id: int,
        model: str,
        temperature: float,
        messages_snapshot: list[dict[str, str]],
    ) -> None:
        super().__init__()
        self.client = client
        self.session_id = session_id
        self.model = model
        self.temperature = temperature
        self.messages_snapshot = messages_snapshot

    @Slot()
    def run(self) -> None:
        try:
            start = time.monotonic()
            reply_parts: list[str] = []
            usage_data = None
            stream = self.client.chat.completions.create(
                messages=self.messages_snapshot,
                model=self.model,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage_data = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    reply_parts.append(content)
                    self.response_delta.emit(self.session_id, content)
            reply = "".join(reply_parts).strip()
            usage = usage_to_dict(usage_data)
            meta = {
                "latency": time.monotonic() - start,
                "usage": usage,
                "model": self.model,
            }
            meta["cost"] = estimate_cost(self.model, usage)
            self.response_ready.emit(self.session_id, reply, meta)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(self.session_id, str(exc))
        finally:
            self.done.emit(self.session_id)


class Avatar(QLabel):
    def __init__(self, text: str, bg: str) -> None:
        super().__init__(text)
        self.setFixedSize(34, 34)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"""
            QLabel {{
                background-color: {bg};
                color: white;
                border-radius: 17px;
                font-weight: 700;
                font-size: 12px;
            }}
            """
        )


class MessageBubble(QFrame):
    def __init__(
        self,
        author: str,
        content: str,
        role: str,
        wrap: int = 700,
        rich_text: bool = False,
        meta_text: str | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("bubble")

        if role == "user":
            bg = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #22d3ee)"
            meta_color = "#dff4ff"
            border = "#38bdf8"
        else:
            bg = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f3d5e, stop:1 #136b97)"
            meta_color = "#cfe9ff"
            border = "#1c6ea4"

        self.setStyleSheet(
            f"""
            QFrame#bubble {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 18px;
                padding: 12px;
            }}
            QLabel {{
                color: #ffffff;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        meta = datetime.now().strftime("%H:%M")
        header = QLabel(f"{author} - {meta}")
        header.setStyleSheet(f"color: {meta_color}; font-weight: 700; font-size: 12px;")
        layout.addWidget(header)

        self.body = QLabel(content)
        self.body.setWordWrap(True)
        self.body.setStyleSheet("font-size: 14px; line-height: 1.4em;")
        self.body.setMaximumWidth(wrap)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setTextFormat(Qt.RichText if rich_text else Qt.PlainText)
        layout.addWidget(self.body)

        self.meta_label = QLabel(meta_text or "")
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet(f"color: {meta_color}; font-size: 11px;")
        self.meta_label.setVisible(bool(meta_text))
        layout.addWidget(self.meta_label)


class SystemNotice(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                color: #a3b5cc;
                background-color: rgba(15, 23, 42, 0.85);
                border: 1px solid #22324a;
                border-radius: 12px;
                padding: 7px 12px;
                font-size: 12px;
            }
            """
        )


class MessageRow(QWidget):
    def __init__(
        self, author: str, content: str, role: str, rich_text: bool = False, meta_text: str | None = None
    ) -> None:
        super().__init__()
        self.body_label: QLabel | None = None
        self.meta_label: QLabel | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        if role == "system":
            layout.addStretch(1)
            layout.addWidget(SystemNotice(content))
            layout.addStretch(1)
            return

        bubble = MessageBubble(author, content, role, rich_text=rich_text, meta_text=meta_text)
        bubble.setMaximumWidth(760)
        self.body_label = bubble.body
        self.meta_label = bubble.meta_label

        if role == "user":
            layout.addStretch(1)
            layout.addWidget(bubble, 0)
            layout.addWidget(Avatar("YOU", "#0ea5e9"), 0)
        else:
            layout.addWidget(Avatar("MA", "#136b97"), 0)
            layout.addWidget(bubble, 0)
            layout.addStretch(1)


class SettingsDialog(QDialog):
    def __init__(self, owner: "ChatWindow") -> None:
        super().__init__(owner)
        self.owner = owner
        self.card: QFrame | None = None
        self.model_selector: QComboBox | None = None
        self.temperature_spin: QDoubleSpinBox | None = None
        self.compact_toggle: QCheckBox | None = None
        self.system_prompt_edit: QPlainTextEdit | None = None
        self.apply_prompt_btn: QPushButton | None = None
        self.close_btn: QPushButton | None = None

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(0)
        root.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("modalCard")
        card.setMinimumWidth(560)
        card.setMaximumWidth(700)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 170))
        card.setGraphicsEffect(shadow)
        self.card = card

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(12)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(10)

        title = QLabel("Configuracion")
        title.setObjectName("modalTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)

        close_btn = QPushButton("Cerrar")
        close_btn.setObjectName("ghost")
        close_btn.clicked.connect(owner.close_settings)
        title_row.addWidget(close_btn)
        self.close_btn = close_btn

        card_layout.addLayout(title_row)

        subtitle = QLabel("Ajusta modelo, prompt y estilo para este chat.")
        subtitle.setObjectName("muted")
        card_layout.addWidget(subtitle)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        card_layout.addWidget(divider)

        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        model_label = QLabel("Modelo")
        model_label.setObjectName("section")
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(260)

        temp_label = QLabel("Temperatura")
        temp_label.setObjectName("section")
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(1)

        self.compact_toggle = QCheckBox("Modo compacto")

        form.addWidget(model_label, 0, 0)
        form.addWidget(self.model_selector, 0, 1)
        form.addWidget(temp_label, 1, 0)
        form.addWidget(self.temperature_spin, 1, 1)
        form.addWidget(self.compact_toggle, 2, 1)

        card_layout.addLayout(form)

        profiles_label = QLabel("Perfiles")
        profiles_label.setObjectName("section")
        card_layout.addWidget(profiles_label)

        profiles_row = QHBoxLayout()
        profiles_row.setContentsMargins(0, 0, 0, 0)
        profiles_row.setSpacing(8)
        for name in PROFILE_PRESETS:
            btn = QPushButton(name)
            btn.setProperty("chip", True)
            btn.clicked.connect(lambda _, n=name: owner.apply_profile(n))
            profiles_row.addWidget(btn)
        profiles_row.addStretch(1)
        card_layout.addLayout(profiles_row)

        prompt_label = QLabel("Prompt del sistema")
        prompt_label.setObjectName("section")
        card_layout.addWidget(prompt_label)

        self.system_prompt_edit = QPlainTextEdit()
        self.system_prompt_edit.setMaximumHeight(140)
        self.system_prompt_edit.setPlaceholderText("Prompt del sistema para esta sesion...")
        self.system_prompt_edit.setTabChangesFocus(True)
        card_layout.addWidget(self.system_prompt_edit)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(10)

        self.apply_prompt_btn = QPushButton("Aplicar prompt")
        self.apply_prompt_btn.setObjectName("primary")
        self.apply_prompt_btn.clicked.connect(owner.apply_system_prompt)
        actions_row.addWidget(self.apply_prompt_btn)

        actions_row.addStretch(1)

        close_btn_2 = QPushButton("Cerrar")
        close_btn_2.setObjectName("ghost")
        close_btn_2.clicked.connect(owner.close_settings)
        actions_row.addWidget(close_btn_2)

        card_layout.addLayout(actions_row)

        root.addWidget(card)

    def mousePressEvent(self, event):  # noqa: N802
        if self.card and not self.card.geometry().contains(event.pos()):
            self.owner.close_settings()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.owner.close_settings()
            return
        super().keyPressEvent(event)


class ChatWindow(QMainWindow):
    def __init__(self, client: Groq) -> None:
        super().__init__()
        self.client = client
        self.sessions: list[dict] = []
        self.sessions_by_id: dict[int, dict] = {}
        self.active_session_id: int | None = None
        self.session_counter = 0
        self.thread: QThread | None = None
        self.worker: ChatWorker | None = None
        self.busy = False
        self._worker_done_session_id: int | None = None
        self._typing_active = False
        self._typing_index = 0
        self._typing_text = ""
        self._typing_target: QLabel | None = None
        self._typing_use_format = False
        self.typing_timer = QTimer(self)
        self.typing_timer.setInterval(20)
        self.typing_timer.timeout.connect(self._typing_tick)
        self.model_selector: QComboBox | None = None
        self.model_label: QLabel | None = None
        self.temperature_spin: QDoubleSpinBox | None = None
        self.compact_toggle: QCheckBox | None = None
        self.system_prompt_edit: QPlainTextEdit | None = None
        self.apply_prompt_btn: QPushButton | None = None
        self.settings_dialog: SettingsDialog | None = None
        self.settings_btn: QPushButton | None = None
        self._active_animations: list[QPropertyAnimation] = []
        self._modal_anim: QPropertyAnimation | None = None
        self._intro_played = False
        self._stream_buffers: dict[int, str] = {}
        self._stream_rows: dict[int, MessageRow] = {}
        self._streaming_sessions: set[int] = set()

        self.setWindowTitle("Monris AI")
        self.resize(1200, 780)
        self.setMinimumSize(980, 640)
        self.build_palette()
        self.build_ui()
        self.build_settings_dialog()
        self.new_chat()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self._intro_played:
            return
        self._intro_played = True
        self.setWindowOpacity(0.0)
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(220)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)

        def cleanup() -> None:
            if anim in self._active_animations:
                self._active_animations.remove(anim)

        anim.finished.connect(cleanup)
        self._active_animations.append(anim)
        anim.start()

    def build_palette(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: qradialgradient(cx:0.2, cy:0.1, radius:1, fx:0.2, fy:0.1,
                    stop:0 #111827, stop:0.6 #0a0f1a, stop:1 #06080f);
                font-family: "Space Grotesk";
            }
            QLabel#brand { font-size: 19px; font-weight: 800; color: #f8fafc; }
            QLabel#title { font-size: 24px; font-weight: 800; color: #f1f5f9; }
            QLabel#muted { color: #9aa7bd; font-size: 12px; }
            QLabel#section { color: #93a4be; font-size: 11px; }
            QLabel#status { color: #93a4be; font-size: 12px; }
            QFrame#sidebar, QFrame#main {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0f1a2d, stop:1 #0b1424);
                border: 1px solid #162a40;
                border-radius: 20px;
            }
            QFrame#chatframe {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0b1424, stop:1 #0a111f);
                border: 1px solid #1e3048;
                border-radius: 18px;
            }
            QFrame#bottom {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0b1527, stop:1 #0c192e);
                border: 1px solid #1d2b41;
                border-radius: 18px;
            }
            QPlainTextEdit {
                background-color: #0f172a;
                color: #e2e8f0;
                border: 1px solid #1f2a3a;
                border-radius: 14px;
                padding: 10px;
                font-size: 14px;
            }
            QPlainTextEdit:focus { border: 1px solid #38bdf8; }
            QPushButton {
                background-color: #182438;
                color: #e2e8f0;
                border: 1px solid #243246;
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #20304a; }
            QPushButton:pressed { background-color: #162235; }
            QPushButton#primary {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #22d3ee);
                color: #06101f;
                border: none;
            }
            QPushButton#primary:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #67e8f9);
            }
            QPushButton#ghost {
                background-color: transparent;
                color: #cbd5e1;
                border: 1px solid #263248;
                padding: 8px 12px;
            }
            QPushButton#ghost:hover { background-color: #122034; }
            QPushButton:disabled { background-color: #334155; }
            QPushButton[chip="true"] {
                background-color: #0f172a;
                border: 1px solid #1f2a3a;
                border-radius: 14px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton[chip="true"]:hover { background-color: #152038; }
            QListWidget {
                background-color: #0b1424;
                border: 1px solid #1b2536;
                border-radius: 12px;
                color: #cbd5e1;
            }
            QListWidget::item { padding: 8px 10px; border-radius: 10px; }
            QListWidget::item:selected { background-color: #1e293b; }
            QListWidget::item:hover { background-color: #182232; }
            QComboBox {
                background-color: #0f172a;
                color: #e2e8f0;
                border: 1px solid #1f2a3a;
                border-radius: 10px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #e2e8f0;
                selection-background-color: #1e293b;
                border: 1px solid #1f2a3a;
            }
            QComboBox:focus { border: 1px solid #38bdf8; }
            QDoubleSpinBox {
                background-color: #0f172a;
                color: #e2e8f0;
                border: 1px solid #1f2a3a;
                border-radius: 10px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 14px; }
            QDoubleSpinBox:focus { border: 1px solid #38bdf8; }
            QCheckBox { color: #cbd5e1; font-size: 12px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
            QCheckBox::indicator:unchecked { background-color: #0f172a; border: 1px solid #2b3b55; }
            QCheckBox::indicator:checked { background-color: #22d3ee; border: 1px solid #22d3ee; }
            QDialog { background-color: rgba(5, 9, 16, 190); }
            QFrame#modalCard {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0b1324, stop:1 #0f1b33);
                border: 1px solid #1f2f47;
                border-radius: 18px;
            }
            QLabel#modalTitle { font-size: 20px; font-weight: 800; color: #f8fafc; }
            QFrame#divider { background-color: #1b2a41; }
            QScrollArea { border: none; background: transparent; }
            QScrollArea::viewport { background: transparent; }
            QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
            QScrollBar::handle:vertical { background: #2b3144; border-radius: 4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            """
        )

    def build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addWidget(self.build_sidebar(), 0)
        root.addWidget(self.build_main_panel(), 1)

        self.setCentralWidget(central)

    def build_sidebar(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setFixedWidth(280)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        brand = QLabel("Monris AI")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        subtitle = QLabel("Powered by Groq")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)

        self.new_btn = QPushButton("New chat")
        self.new_btn.setObjectName("primary")
        self.new_btn.clicked.connect(self.new_chat)
        layout.addWidget(self.new_btn)

        self.settings_btn = QPushButton("Configuracion")
        self.settings_btn.setObjectName("ghost")
        self.settings_btn.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_btn)

        section = QLabel("Recent chats")
        section.setObjectName("section")
        layout.addWidget(section)

        self.history = QListWidget()
        self.history.currentItemChanged.connect(self.on_history_selected)
        layout.addWidget(self.history, 1)

        status_title = QLabel("Activo")
        status_title.setObjectName("section")
        layout.addWidget(status_title)

        self.model_label = QLabel("")
        self.model_label.setObjectName("muted")
        layout.addWidget(self.model_label)
        self._refresh_model_status()

        return panel

    def build_settings_dialog(self) -> None:
        self.settings_dialog = SettingsDialog(self)
        dialog = self.settings_dialog
        self.model_selector = dialog.model_selector
        self.temperature_spin = dialog.temperature_spin
        self.compact_toggle = dialog.compact_toggle
        self.system_prompt_edit = dialog.system_prompt_edit
        self.apply_prompt_btn = dialog.apply_prompt_btn

        if self.model_selector:
            self.model_selector.addItems(MODEL_OPTIONS)
            default_model = "llama-3.3-70b-versatile"
            if default_model in MODEL_OPTIONS:
                self.model_selector.setCurrentText(default_model)
            self.model_selector.currentTextChanged.connect(self._on_model_changed)

        if self.temperature_spin:
            self.temperature_spin.setValue(DEFAULT_TEMPERATURE)
            self.temperature_spin.valueChanged.connect(self._on_temperature_changed)

        if self.compact_toggle:
            self.compact_toggle.toggled.connect(self._on_compact_toggled)

        if self.system_prompt_edit:
            self.system_prompt_edit.setPlainText(SYSTEM_PROMPT)

        self._refresh_model_status()

    def open_settings(self) -> None:
        if not self.settings_dialog:
            return
        self.settings_dialog.setWindowOpacity(0.0)
        self.settings_dialog.setGeometry(self.geometry())
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()
        self._animate_modal(1.0)

    def close_settings(self) -> None:
        if not self.settings_dialog or not self.settings_dialog.isVisible():
            return
        self._animate_modal(0.0)

    def _animate_modal(self, target_opacity: float) -> None:
        if not self.settings_dialog:
            return
        self._modal_anim = QPropertyAnimation(self.settings_dialog, b"windowOpacity", self.settings_dialog)
        self._modal_anim.setDuration(180)
        self._modal_anim.setStartValue(self.settings_dialog.windowOpacity())
        self._modal_anim.setEndValue(target_opacity)
        self._modal_anim.setEasingCurve(QEasingCurve.OutCubic)
        if target_opacity == 0.0:
            self._modal_anim.finished.connect(self.settings_dialog.hide)
        self._modal_anim.start()

    def build_main_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("main")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        layout.addLayout(self.build_header())
        layout.addWidget(self.build_chat_frame(), 1)
        layout.addWidget(self.build_typing_line())
        layout.addWidget(self.build_bottom_panel())

        return panel

    def build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.chat_title = QLabel("New chat")
        self.chat_title.setObjectName("title")
        row.addWidget(self.chat_title)

        pill = QLabel("Online")
        pill.setStyleSheet(
            """
            QLabel {
                background-color: #0f2740;
                color: #67e8f9;
                border: 1px solid #1b3551;
                border-radius: 10px;
                padding: 4px 8px;
                font-size: 11px;
            }
            """
        )
        row.addWidget(pill)

        row.addStretch(1)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        row.addWidget(self.status_label)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("ghost")
        self.clear_btn.clicked.connect(self.clear_chat)
        row.addWidget(self.clear_btn)

        return row

    def build_chat_frame(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("chatframe")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        container = QScrollArea()
        container.setWidgetResizable(True)
        container.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.chat_widget = QWidget()
        self.chat_widget.setStyleSheet("background: transparent;")
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setContentsMargins(6, 6, 6, 6)
        self.chat_layout.setSpacing(12)
        self.chat_layout.addStretch(1)

        container.setWidget(self.chat_widget)
        layout.addWidget(container)
        return frame

    def build_typing_line(self) -> QLabel:
        self.typing_label = QLabel("")
        self.typing_label.setObjectName("muted")
        self.typing_label.setVisible(False)
        return self.typing_label

    def build_bottom_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("bottom")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        layout.addLayout(self.build_quick_actions())
        layout.addLayout(self.build_input_row())

        return panel

    def build_quick_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        tips = [
            "Summarize in 3 bullets",
            "Explain step by step",
            "Give me a short answer",
        ]
        for text in tips:
            btn = QPushButton(text)
            btn.setProperty("chip", True)
            btn.clicked.connect(lambda _, t=text: self.fill_prompt(t))
            row.addWidget(btn)

        row.addStretch(1)
        return row

    def build_input_row(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        hint = QLabel("Enter to send - Shift+Enter for new line")
        hint.setObjectName("muted")
        col.addWidget(hint)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.prompt_box = QPlainTextEdit()
        self.prompt_box.setPlaceholderText("Type your message...")
        self.prompt_box.setMaximumHeight(140)
        self.prompt_box.setTabChangesFocus(True)
        self.prompt_box.installEventFilter(self)
        row.addWidget(self.prompt_box, 1)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.setMinimumWidth(120)
        self.send_btn.clicked.connect(self.send)
        row.addWidget(self.send_btn, 0)

        col.addLayout(row)
        return col

    def _default_messages(self, system_prompt: str | None = None) -> list[dict[str, str]]:
        prompt = (system_prompt or SYSTEM_PROMPT).strip()
        if not prompt:
            prompt = SYSTEM_PROMPT
        return [{"role": "system", "content": prompt}]

    def current_system_prompt(self) -> str:
        if self.system_prompt_edit:
            prompt = self.system_prompt_edit.toPlainText().strip()
            if prompt:
                return prompt
        return SYSTEM_PROMPT

    def current_temperature(self) -> float:
        if self.temperature_spin:
            return float(self.temperature_spin.value())
        return DEFAULT_TEMPERATURE

    def _refresh_model_status(self) -> None:
        if not self.model_label:
            return
        model = self.current_model()
        temp = self.current_temperature()
        model_text = model or "N/A"
        self.model_label.setText(f"Modelo: {model_text} | Temp: {temp:.1f}")

    def _on_temperature_changed(self, value: float) -> None:
        session = self._current_session()
        if session:
            session["temperature"] = float(value)
        self._refresh_model_status()

    def _ensure_system_message(self, session: dict, prompt: str) -> None:
        if session["messages"] and session["messages"][0]["role"] == "system":
            session["messages"][0]["content"] = prompt
            return
        session["messages"].insert(0, {"role": "system", "content": prompt})

    def _messages_for_api(self, session: dict) -> list[dict[str, str]]:
        snapshot: list[dict[str, str]] = []
        for msg in session["messages"]:
            role = msg.get("role")
            content = msg.get("content")
            if role and content is not None:
                snapshot.append({"role": role, "content": content})
        return snapshot

    def apply_system_prompt(self) -> None:
        session = self._current_session()
        if not session or not self.system_prompt_edit:
            return
        prompt = self.system_prompt_edit.toPlainText().strip()
        if not prompt:
            prompt = SYSTEM_PROMPT
            self.system_prompt_edit.setPlainText(prompt)
        session["system_prompt"] = prompt
        self._ensure_system_message(session, prompt)
        self.flash_status("Prompt aplicado")

    def apply_profile(self, name: str) -> None:
        preset = PROFILE_PRESETS.get(name)
        if not preset:
            return
        if self.system_prompt_edit:
            self.system_prompt_edit.setPlainText(preset["prompt"])
        if self.temperature_spin:
            self.temperature_spin.setValue(preset["temperature"])
        self.apply_system_prompt()

    def _on_compact_toggled(self, _checked: bool) -> None:
        session = self._current_session()
        if session:
            self.render_session(session)

    def current_model(self) -> str:
        if self.model_selector:
            current = self.model_selector.currentText().strip()
            if current:
                return current
        if "llama-3.3-70b-versatile" in MODEL_OPTIONS:
            return "llama-3.3-70b-versatile"
        return MODEL_OPTIONS[0] if MODEL_OPTIONS else ""

    def _on_model_changed(self, model: str) -> None:
        session = self._current_session()
        if session:
            session["model"] = model
        self._refresh_model_status()

    def _title_from_prompt(self, prompt: str) -> str:
        cleaned = " ".join(prompt.split())
        if not cleaned:
            return "New chat"
        if len(cleaned) > 28:
            return f"{cleaned[:28].rstrip()}..."
        return cleaned

    def _current_session(self) -> dict | None:
        if self.active_session_id is None:
            return None
        return self.sessions_by_id.get(self.active_session_id)

    def _create_session(self) -> dict:
        self.session_counter += 1
        session_id = self.session_counter
        title = f"Chat {session_id}"
        item = QListWidgetItem(title)
        item.setData(Qt.UserRole, session_id)
        self.history.insertItem(0, item)

        system_prompt = self.current_system_prompt()
        session = {
            "id": session_id,
            "title": title,
            "messages": self._default_messages(system_prompt),
            "item": item,
            "auto_title": True,
            "model": self.current_model(),
            "temperature": self.current_temperature(),
            "system_prompt": system_prompt,
        }
        self.sessions.insert(0, session)
        self.sessions_by_id[session_id] = session
        return session

    def new_chat(self) -> None:
        session = self._create_session()
        self.set_active_session(session["id"])

    def set_active_session(self, session_id: int) -> None:
        session = self.sessions_by_id.get(session_id)
        if not session:
            return
        self.active_session_id = session_id
        self.chat_title.setText(session["title"])
        if self.model_selector and session.get("model"):
            self.model_selector.setCurrentText(session["model"])
        if self.temperature_spin and session.get("temperature") is not None:
            self.temperature_spin.blockSignals(True)
            self.temperature_spin.setValue(session["temperature"])
            self.temperature_spin.blockSignals(False)
        if self.system_prompt_edit and session.get("system_prompt"):
            self.system_prompt_edit.setPlainText(session["system_prompt"])
        self._refresh_model_status()

        if self.history.currentItem() is not session["item"]:
            self.history.blockSignals(True)
            self.history.setCurrentItem(session["item"])
            self.history.blockSignals(False)

        self.render_session(session)

    def on_history_selected(self, current: QListWidgetItem, _previous: QListWidgetItem) -> None:
        if not current:
            return
        session_id = current.data(Qt.UserRole)
        self.set_active_session(session_id)

    def promote_session(self, session: dict) -> None:
        item = session.get("item")
        if not item:
            return
        row = self.history.row(item)
        if row > 0:
            self.history.takeItem(row)
            self.history.insertItem(0, item)
            self.history.setCurrentItem(item)
        if self.sessions and self.sessions[0] is not session:
            self.sessions.remove(session)
            self.sessions.insert(0, session)

    def render_session(self, session: dict, notice: str | None = None) -> None:
        self._stop_typing_animation()
        self._clear_layout(self.chat_layout)
        self._stream_rows.clear()
        self.chat_layout.addStretch(1)

        chat_messages = [msg for msg in session["messages"] if msg["role"] in ("user", "assistant")]
        has_chat = bool(chat_messages)
        if notice:
            self.add_message("System", notice, role="system", scroll=False, animate=False)
        if not has_chat:
            self.add_message("System", "Start a new conversation.", role="system", scroll=False, animate=False)
        else:
            messages_to_show = chat_messages
            hidden_count = 0
            if self.compact_toggle and self.compact_toggle.isChecked():
                if len(messages_to_show) > COMPACT_MAX_MESSAGES:
                    hidden_count = len(messages_to_show) - COMPACT_MAX_MESSAGES
                    messages_to_show = messages_to_show[-COMPACT_MAX_MESSAGES:]
            if hidden_count:
                self.add_message(
                    "System",
                    f"Compact mode: {hidden_count} messages hidden. Turn off compact to see all.",
                    role="system",
                    scroll=False,
                    animate=False,
                )
            for msg in messages_to_show:
                role = msg["role"]
                if role == "user":
                    author = "You"
                elif role == "assistant":
                    author = "Monris AI"
                else:
                    continue
                self.add_message(
                    author, msg["content"], role=role, scroll=False, meta=msg.get("meta"), animate=False
                )

            if session["id"] in self._streaming_sessions:
                partial = self._stream_buffers.get(session["id"], "")
                if partial:
                    row = self.add_message("Monris AI", partial, role="assistant", scroll=False, animate=False)
                    if row.meta_label:
                        row.meta_label.setText("Streaming...")
                        row.meta_label.setVisible(True)
                    self._stream_rows[session["id"]] = row

        QTimer.singleShot(0, self.scroll_to_bottom)

    def fill_prompt(self, text: str) -> None:
        self.prompt_box.setPlainText(text)
        self.prompt_box.setFocus()
        self.prompt_box.moveCursor(self.prompt_box.textCursor().End)

    def add_message(
        self,
        author: str,
        content: str,
        role: str,
        scroll: bool = True,
        rich_text: bool | None = None,
        meta: dict | None = None,
        animate: bool = True,
    ) -> MessageRow:
        if role == "assistant":
            content = format_assistant_message(content)
            if rich_text is None:
                rich_text = True
        elif rich_text is None:
            rich_text = False

        meta_text = format_meta_text(meta) if meta else None
        row = MessageRow(author, content, role, rich_text=rich_text, meta_text=meta_text)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        if animate:
            self._animate_message(row)
        if scroll:
            QTimer.singleShot(0, self.scroll_to_bottom)
        return row

    def scroll_to_bottom(self) -> None:
        area: QScrollArea = self.centralWidget().findChild(QScrollArea)
        if area:
            area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())

    def _animate_message(self, row: QWidget) -> None:
        effect = QGraphicsOpacityEffect(row)
        effect.setOpacity(0.0)
        row.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", row)
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)

        def cleanup() -> None:
            row.setGraphicsEffect(None)
            if anim in self._active_animations:
                self._active_animations.remove(anim)

        anim.finished.connect(cleanup)
        self._active_animations.append(anim)
        anim.start()

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.send_btn.setDisabled(busy)
        self.prompt_box.setDisabled(busy)
        self.status_label.setText("Thinking..." if busy else "Ready")
        self.typing_label.setText("Monris AI is typing..." if busy else "")
        self.typing_label.setVisible(busy)

    def flash_status(self, text: str, duration_ms: int = 1600) -> None:
        if self.busy:
            return
        self.status_label.setText(text)
        QTimer.singleShot(duration_ms, self._restore_status)

    def _restore_status(self) -> None:
        self.status_label.setText("Thinking..." if self.busy else "Ready")

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def clear_chat(self) -> None:
        session = self._current_session()
        if not session:
            return
        session["messages"] = self._default_messages(session.get("system_prompt"))
        self.render_session(session, notice="Chat reset.")

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.prompt_box and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
                self.send()
                return True
        return super().eventFilter(obj, event)

    def send(self) -> None:
        if self.busy:
            return
        session = self._current_session()
        if not session:
            return
        self.apply_system_prompt()
        text = self.prompt_box.toPlainText().strip()
        if not text:
            return
        self.prompt_box.clear()
        self.add_message("You", text, role="user")
        session["messages"].append({"role": "user", "content": text})

        if session.get("auto_title"):
            new_title = self._title_from_prompt(text)
            session["title"] = new_title
            session["auto_title"] = False
            session["item"].setText(new_title)
            self.chat_title.setText(new_title)

        self.promote_session(session)
        self.start_worker(session["id"], self._messages_for_api(session))

    @Slot(int)
    def on_done(self, _session_id: int) -> None:
        self._worker_done_session_id = _session_id
        self._maybe_finish_busy()

    @Slot(int, str)
    def on_response_delta(self, session_id: int, delta: str) -> None:
        buffer = self._stream_buffers.get(session_id, "") + delta
        self._stream_buffers[session_id] = buffer
        if session_id != self.active_session_id:
            return
        row = self._stream_rows.get(session_id)
        if not row:
            row = self.add_message("Monris AI", buffer, role="assistant")
            self._stream_rows[session_id] = row
        if row.body_label:
            row.body_label.setText(format_assistant_message(buffer))
        if row.meta_label:
            row.meta_label.setText("Streaming...")
            row.meta_label.setVisible(True)

    @Slot(int, str, dict)
    def on_response(self, session_id: int, msg: str, meta: dict) -> None:
        session = self.sessions_by_id.get(session_id)
        if not session:
            return
        session["messages"].append({"role": "assistant", "content": msg, "meta": meta})
        self._streaming_sessions.discard(session_id)
        self._stream_buffers.pop(session_id, None)
        row = self._stream_rows.pop(session_id, None)
        if session_id == self.active_session_id:
            if row and row.body_label:
                row.body_label.setText(format_assistant_message(msg))
                if row.meta_label:
                    row.meta_label.setText(format_meta_text(meta))
                    row.meta_label.setVisible(True)
            else:
                self.add_message("Monris AI", msg, role="assistant", meta=meta)
            QTimer.singleShot(0, self.scroll_to_bottom)
        self.promote_session(session)

    @Slot(int, str)
    def on_error(self, _session_id: int, msg: str) -> None:
        self._streaming_sessions.discard(_session_id)
        self._stream_buffers.pop(_session_id, None)
        row = self._stream_rows.pop(_session_id, None)
        if row and row.meta_label:
            row.meta_label.setText("Error")
            row.meta_label.setVisible(True)
        QMessageBox.critical(self, "Error", msg)

    def start_worker(self, session_id: int, messages_snapshot: list[dict[str, str]]) -> None:
        self._worker_done_session_id = None
        self._stop_typing_animation()
        self.set_busy(True)
        session = self.sessions_by_id.get(session_id)
        model = session.get("model") if session else self.current_model()
        temperature = session.get("temperature") if session else self.current_temperature()
        if temperature is None:
            temperature = self.current_temperature()
        self._streaming_sessions.add(session_id)
        self._stream_buffers.pop(session_id, None)
        self._stream_rows.pop(session_id, None)
        self.thread = QThread()
        self.worker = ChatWorker(self.client, session_id, model, temperature, messages_snapshot)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.response_delta.connect(self.on_response_delta, Qt.QueuedConnection)
        self.worker.response_ready.connect(self.on_response, Qt.QueuedConnection)
        self.worker.error.connect(self.on_error, Qt.QueuedConnection)
        self.worker.done.connect(self.on_done, Qt.QueuedConnection)
        self.worker.done.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _start_typing_message(self, author: str, content: str, role: str) -> None:
        row = self.add_message(author, "", role=role)
        if not row.body_label:
            return
        self._typing_text = content
        self._typing_index = 0
        self._typing_target = row.body_label
        self._typing_use_format = role == "assistant"
        self._typing_active = True
        if self.typing_timer.isActive():
            self.typing_timer.stop()
        self.typing_timer.start()
        self._typing_tick()

    def _typing_tick(self) -> None:
        if not self._typing_active or not self._typing_target:
            self._stop_typing_animation()
            return
        if self._typing_index >= len(self._typing_text):
            if self._typing_use_format:
                self._typing_target.setText(format_assistant_message(self._typing_text))
            else:
                self._typing_target.setText(self._typing_text)
            self._stop_typing_animation()
            QTimer.singleShot(0, self.scroll_to_bottom)
            return

        step = 2
        current_char = self._typing_text[self._typing_index]
        if current_char in ".!?\n":
            step = 1

        self._typing_index = min(len(self._typing_text), self._typing_index + step)
        visible_text = self._typing_text[: self._typing_index]
        if self._typing_use_format:
            self._typing_target.setText(format_assistant_message(visible_text))
        else:
            self._typing_target.setText(visible_text)
        if self._typing_index % 12 == 0:
            QTimer.singleShot(0, self.scroll_to_bottom)

    def _stop_typing_animation(self) -> None:
        if self.typing_timer.isActive():
            self.typing_timer.stop()
        self._typing_active = False
        self._typing_index = 0
        self._typing_text = ""
        self._typing_target = None
        self._typing_use_format = False
        self._maybe_finish_busy()

    def _maybe_finish_busy(self) -> None:
        if self._worker_done_session_id is None:
            return
        if not self._typing_active:
            self.set_busy(False)
            self._worker_done_session_id = None


def main() -> None:
    app = QApplication(sys.argv)

    try:
        client = build_client()
    except RuntimeError as exc:
        QMessageBox.critical(None, "Configuration error", str(exc))
        return

    window = ChatWindow(client)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

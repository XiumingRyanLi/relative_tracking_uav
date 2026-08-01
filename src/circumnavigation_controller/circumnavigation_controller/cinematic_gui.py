#!/usr/bin/env python3
"""
cinematic_gui.py

Standalone GUI for building CinematicPlanner action sequences and sending
them to relative_position_controller over /cinematic_command (std_msgs/String,
JSON-encoded list of action dicts).

The form is generated entirely from cinematic_action_schema.py -- adding or
changing a field means editing the schema once, not this file. No LLM, no
RViz plugin machinery: a plain PyQt5 form whose widgets map onto the same
schema the controller validates against.

Run:
    ros2 run circumnavigation_controller cinematic_gui
    # or directly:
    python3 cinematic_gui.py
"""
import sys
import json
import signal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QDoubleSpinBox, QPushButton, QListWidget, QLabel,
    QGroupBox, QMessageBox,
)

try:
    from .cinematic_action_schema import (
        ACTION_SCHEMA, ACTION_TYPES, FIELD_SPECS, FIELD_LABELS,
        FIELD_ORDER, LOCATIONS,
    )
except ImportError:
    from cinematic_action_schema import (
        ACTION_SCHEMA, ACTION_TYPES, FIELD_SPECS, FIELD_LABELS,
        FIELD_ORDER, LOCATIONS,
    )


class CinematicGuiNode(Node):
    """Thin ROS2 node: only publishes. Kept separate from the Qt widget so
    the widget doesn't need to know anything about rclpy internals."""

    def __init__(self):
        super().__init__("cinematic_gui")
        self.declare_parameter("cinematic_command_topic", "/cinematic_command")
        topic = self.get_parameter("cinematic_command_topic").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(String, topic, qos)
        self.topic = topic
        self.get_logger().info(f"Cinematic GUI publishing on {topic}")

    def send_sequence(self, actions: list):
        msg = String()
        msg.data = json.dumps(actions)
        self.pub.publish(msg)


class CinematicWindow(QWidget):
    def __init__(self, ros_node: CinematicGuiNode):
        super().__init__()
        self.ros_node = ros_node
        self.queue = []

        self.setWindowTitle("Cinematic Shot Planner")
        self.resize(420, 560)

        root = QVBoxLayout(self)

        builder_box = QGroupBox("Add action")
        self.form = QFormLayout()

        self.type_combo = QComboBox()
        self.type_combo.addItems(ACTION_TYPES)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.form.addRow("Type", self.type_combo)

        # Build one widget per field in FIELD_ORDER, driven entirely by
        # FIELD_SPECS -- no per-field hand-written widget code.
        self._field_widgets = {}
        for field in FIELD_ORDER:
            widget = self._make_widget(FIELD_SPECS[field])
            self.form.addRow(FIELD_LABELS[field], widget)
            self._field_widgets[field] = widget

        builder_box.setLayout(self.form)
        root.addWidget(builder_box)

        add_btn = QPushButton("Add to queue")
        add_btn.clicked.connect(self._add_action)
        root.addWidget(add_btn)

        root.addWidget(QLabel("Queued sequence:"))
        self.queue_list = QListWidget()
        root.addWidget(self.queue_list)

        queue_btns = QHBoxLayout()
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn = QPushButton("Clear queue")
        clear_btn.clicked.connect(self._clear_queue)
        queue_btns.addWidget(remove_btn)
        queue_btns.addWidget(clear_btn)
        root.addLayout(queue_btns)

        send_btn = QPushButton("Send sequence to controller")
        send_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        send_btn.clicked.connect(self._send_sequence)
        root.addWidget(send_btn)

        self._on_type_changed(self.type_combo.currentText())

    # -- widget construction, driven by FIELD_SPECS --
    def _make_widget(self, spec: dict):
        if spec["kind"] == "location":
            w = QComboBox()
            w.addItems(sorted(LOCATIONS))
            w.setCurrentText(spec["default"])
            return w
        if spec["kind"] == "enum":
            w = QComboBox()
            w.addItems(list(spec["options"]))
            w.setCurrentText(spec["default"])
            return w
        # float
        lo, hi = spec["bounds"]
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(spec["default"])
        w.setSuffix(spec.get("unit", ""))
        w.setSingleStep(0.5)
        return w

    def _on_type_changed(self, action_type: str):
        visible_fields = set(ACTION_SCHEMA[action_type]["fields"])
        for field, widget in self._field_widgets.items():
            show = field in visible_fields
            widget.setVisible(show)
            row_label = self.form.labelForField(widget)
            if row_label is not None:
                row_label.setVisible(show)

        # Nice-to-have: reset duration to this action type's sensible
        # default (a 5s hold vs. a 15s orbit) rather than whatever was
        # left over from the previous type.
        if "duration" in visible_fields:
            default_duration = ACTION_SCHEMA[action_type].get(
                "duration_default", FIELD_SPECS["duration"]["default"]
            )
            self._field_widgets["duration"].setValue(default_duration)

    # -- queue management --
    def _add_action(self):
        action_type = self.type_combo.currentText()
        action = {"type": action_type}
        for field in ACTION_SCHEMA[action_type]["fields"]:
            widget = self._field_widgets[field]
            action[field] = widget.currentText() if isinstance(widget, QComboBox) else widget.value()

        self.queue.append(action)
        self.queue_list.addItem(self._describe(action))

    def _remove_selected(self):
        row = self.queue_list.currentRow()
        if row >= 0:
            self.queue_list.takeItem(row)
            del self.queue[row]

    def _clear_queue(self):
        self.queue.clear()
        self.queue_list.clear()

    def _send_sequence(self):
        if not self.queue:
            QMessageBox.warning(self, "Empty queue", "Add at least one action first.")
            return
        self.ros_node.send_sequence(self.queue)
        QMessageBox.information(
            self, "Sent", f"Sent {len(self.queue)} action(s) to {self.ros_node.topic}."
        )

    @staticmethod
    def _describe(action: dict) -> str:
        parts = [f"{k}={v}" for k, v in action.items() if k != "type"]
        return f"{action['type']}: " + ", ".join(parts)


def main(args=None):
    rclpy.init(args=args)
    ros_node = CinematicGuiNode()

    app = QApplication(sys.argv)
    window = CinematicWindow(ros_node)
    window.show()

    # PyQt installs its own SIGINT handler that just gets ignored while the
    # C++ event loop is running, so Ctrl+C normally does nothing. Restore
    # Python's default handler so SIGINT actually raises KeyboardInterrupt --
    # but that interrupt can still only be delivered when the interpreter
    # gets a chance to run, which the spin_timer below provides every 50ms.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Pump rclpy alongside the Qt event loop so the publisher/params stay
    # healthy without blocking the GUI thread. This timer firing every
    # 50ms is also what lets Python notice a pending SIGINT and act on it --
    # without some periodic callback, the interpreter never regains control
    # long enough to process the signal at all.
    spin_timer = QTimer()
    spin_timer.timeout.connect(lambda: rclpy.spin_once(ros_node, timeout_sec=0.0))
    spin_timer.start(50)

    try:
        app.exec_()
    except KeyboardInterrupt:
        ros_node.get_logger().info("Ctrl+C received, shutting down.")
    finally:
        app.quit()
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
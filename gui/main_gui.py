import sys
import os
import cv2 as cv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QSpinBox, QDoubleSpinBox, QLineEdit, QGroupBox, QFormLayout
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
import threading
import time


sys.path.append(os.path.dirname(__file__))
from utils import params, server_control, client_control, server_frame_queue, log_lock, stats, stats_lock
from server import run_server_gui
from client import run_client_gui

class LogEmitter(QObject):
    log_signal = pyqtSignal(str)

class DroneGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Vector Stream — Настройки и Мониторинг")
        self.resize(1600, 900)

        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.append_log)

        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setText("Видео с сервера появится здесь\n(запустите сервер и клиент)")

        
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(10, 10, 10, 10)

        
        self.btn_start_server = QPushButton("▶ Запустить Сервер")
        self.btn_start_client = QPushButton("▶ Запустить Клиент")
        self.btn_stop_all = QPushButton("⏹ Остановить Всё")

        self.btn_start_server.clicked.connect(self.start_server)
        self.btn_start_client.clicked.connect(self.start_client)
        self.btn_stop_all.clicked.connect(self.stop_all)

        for btn in [self.btn_start_server, self.btn_start_client, self.btn_stop_all]:
            btn.setFixedHeight(40)

        control_layout.addWidget(self.btn_start_server)
        control_layout.addWidget(self.btn_start_client)
        control_layout.addWidget(self.btn_stop_all)

        # Группа параметров клиента
        client_group = QGroupBox("Параметры Клиента")
        client_form = QFormLayout()
        self.widgets = {}


        int_params = [
            ("Кадр каждые N", "SEND_EVERY_N_FRAMES", 1, 100),
            ("Макс. примитивов", "MAX_PRIMITIVES", 1, 20),
            ("Мин. площадь контура", "MIN_CONTOUR_AREA", 10, 5000),
        ]
        float_params = [
            ("Упрощение контура", "EPSILON_FACTOR", 0.001, 0.1),
        ]
        str_params = [
            ("Путь к видео", "VIDEO_PATH"),
            ("IP сервера", "HOST_IP"),
            ("Порт", "PORT"),
        ]

        for label, key, minv, maxv in int_params:
            sb = QSpinBox()
            sb.setRange(minv, maxv)
            sb.setValue(params[key])
            sb.valueChanged.connect(lambda v, k=key: self.update_param(k, v))
            client_form.addRow(label, sb)
            self.widgets[key] = sb

        for label, key, minv, maxv in float_params:
            dsb = QDoubleSpinBox()
            dsb.setRange(minv, maxv)
            dsb.setSingleStep(0.001)
            dsb.setValue(params[key])
            dsb.valueChanged.connect(lambda v, k=key: self.update_param(k, v))
            client_form.addRow(label, dsb)
            self.widgets[key] = dsb

        for label, key in str_params:
            le = QLineEdit(str(params[key]))  
            le.textChanged.connect(lambda t, k=key: self.update_param(k, t))
            client_form.addRow(label, le)
            self.widgets[key] = le

        client_group.setLayout(client_form)
        control_layout.addWidget(client_group)


        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setMaximumHeight(200)
        control_layout.addWidget(QLabel("Лог:"))
        control_layout.addWidget(self.log_text)

        self.stats_label = QLabel("📡 Битрейт: — | 🖼️ FPS: — | 📦 Примитивов: —")
        self.stats_label.setFont(QFont("Monospace", 10))
        self.stats_label.setStyleSheet("background-color: #f0f0f0; padding: 6px; border: 1px solid #ccc;")
        control_layout.addWidget(self.stats_label)

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats_display)
        self.stats_timer.start(500)  

        main_layout.addWidget(self.video_label, 2)
        main_layout.addWidget(control_widget, 1)

        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)  


        self.original_stdout = sys.stdout
        sys.stdout = self

    def update_stats_display(self):
        try:
            with stats_lock:
                br = stats["bitrate_mbps"]
                fps = stats["fps"]
                prim = stats["primitive_count"]
            text = f"📡 Битрейт: {br:.2f} Мбит/с | 🖼️ FPS: {fps:.1f} | 📦 Примитивов: {prim}"
            self.stats_label.setText(text)
        except:
            pass

    def write(self, text):
        if text.strip():
            self.log_emitter.log_signal.emit(text.strip())
        self.original_stdout.write(text)

    def flush(self):
        self.original_stdout.flush()

    def append_log(self, text):
        self.log_text.append(text)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def update_param(self, key, value):
        try:
            
            if key in ["PORT", "SEND_EVERY_N_FRAMES", "MAX_PRIMITIVES", "MIN_CONTOUR_AREA"]:
                params[key] = int(value)
            elif key in ["EPSILON_FACTOR"]:
                params[key] = float(value)
            else:
                
                params[key] = value
        except ValueError:
            
            pass

    def start_server(self):
        if not server_control.is_set():
            server_control.set()
            threading.Thread(target=run_server_gui, daemon=True).start()
            self.btn_start_server.setText("⏸ Сервер Запущен")

    def start_client(self):
        if not client_control.is_set():
            client_control.set()
            threading.Thread(target=run_client_gui, daemon=True).start()
            self.btn_start_client.setText("⏸ Клиент Запущен")

    def stop_all(self):
        client_control.clear()
        server_control.clear()
        self.btn_start_server.setText("▶ Запустить Сервер")
        self.btn_start_client.setText("▶ Запустить Клиент")
        self.video_label.setText("Остановлено")

    def update_video(self):
        try:
            if not server_frame_queue.empty():
                frame = server_frame_queue.get_nowait()
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(
                    QPixmap.fromImage(q_img).scaled(
                        self.video_label.width(),
                        self.video_label.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )
        except Exception as e:

            pass

    def closeEvent(self, event):
        self.stop_all()
        sys.stdout = self.original_stdout
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneGUI()
    window.show()
    sys.exit(app.exec())
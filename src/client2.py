import socket
import cv2
import time
import struct
import numpy as np
from threading import Thread, Event
import matplotlib.pyplot as plt


class VideoClient:
    def __init__(self, host, port=5000):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Параметры потока (0.1 Мбит/с = 100 000 бит/с)
        self.target_bitrate = 100000  # бит/сек
        self.target_fps = 25
        self.frame_interval = 1.0 / self.target_fps
        self.width, self.height = (
            720,
            600,
        )  # Уменьшенное разрешение для битрейта 0.1 Мбит/с

        # Статистика
        self.bitrate_history = []
        self.fps_history = []
        self.frame_sizes = []
        self.start_time = None
        self.frame_count = 0
        self.running = Event()
        self.running.set()

        # Кодирование
        self.jpeg_quality = 10  # Стартовое качество
        self.max_quality = 20
        self.min_quality = 5
        self.quality_history = []

        # Ограничитель битрейта
        self.token_bucket = self.target_bitrate / 8  # Байт/сек
        self.bucket_capacity = self.token_bucket
        self.last_update = time.time()

    def connect(self):
        self.sock.connect((self.host, self.port))
        print(f"[+] Подключено к серверу {self.host}:{self.port}")

    def update_token_bucket(self):
        """Обновление токенов для ограничения битрейта"""
        current_time = time.time()
        elapsed = current_time - self.last_update
        tokens_to_add = elapsed * self.token_bucket
        self.bucket_capacity = min(
            self.bucket_capacity + tokens_to_add, self.token_bucket
        )
        self.last_update = current_time

    def consume_tokens(self, frame_size):
        """Потребление токенов и ожидание при необходимости"""
        while self.bucket_capacity < frame_size:
            time.sleep(0.001)
            self.update_token_bucket()
        self.bucket_capacity -= frame_size

    def start_statistics_thread(self):
        """Поток для сбора и отображения статистики"""

        def statistics_monitor():
            last_count = 0
            last_time = time.time()

            while self.running.is_set():
                time.sleep(1.0)
                current_count = self.frame_count
                current_time = time.time()
                elapsed = current_time - last_time
                fps = (current_count - last_count) / elapsed if elapsed > 0 else 0

                if len(self.frame_sizes) > 0:
                    total_bytes = sum(self.frame_sizes)
                    bitrate = (total_bytes * 8) / elapsed  # бит/сек
                else:
                    bitrate = 0

                self.fps_history.append(fps)
                self.bitrate_history.append(bitrate)
                self.quality_history.append(self.jpeg_quality)

                # Адаптация качества
                if bitrate > 0 and len(self.frame_sizes) > 0:
                    avg_frame_size = total_bytes / len(self.frame_sizes)

                    if bitrate > self.target_bitrate * 1.5:
                        self.jpeg_quality = max(self.min_quality, self.jpeg_quality - 3)
                    elif bitrate > self.target_bitrate * 1.2:
                        self.jpeg_quality = max(self.min_quality, self.jpeg_quality - 1)
                    elif bitrate < self.target_bitrate * 0.8:
                        self.jpeg_quality = min(self.max_quality, self.jpeg_quality + 1)

                # Отображение в Мбит/с (как просили)
                print(
                    f"\n[СТАТИСТИКА] FPS: {fps:.1f}/{self.target_fps} | "
                    f"Битрейт: {bitrate / 1000000:.3f} Мбит/с (цель: {self.target_bitrate / 1000000:.3f}) | "
                    f"Качество: {self.jpeg_quality} | "
                    f"Размер кадра: {avg_frame_size:.0f} байт"
                )

                last_count = current_count
                last_time = current_time
                self.frame_sizes = []

        Thread(target=statistics_monitor, daemon=True).start()

    def start_stream(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")

        # Установка параметров камеры
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Форсирование MJPEG и минимального буфера
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"Настройки камеры: {self.width}x{self.height} @ {self.target_fps} FPS")
        self.start_time = time.time()
        self.last_update = time.time()
        self.start_statistics_thread()

        try:
            while self.running.is_set():
                frame_start = time.perf_counter()

                # Захват кадра
                ret, frame = cap.read()
                if not ret:
                    print("Ошибка захвата кадра")
                    break

                # Кодирование в JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                _, buffer = cv2.imencode(".jpg", frame, encode_param)
                data = buffer.tobytes()
                frame_size = len(data)

                # Ограничение битрейта
                self.update_token_bucket()
                self.consume_tokens(frame_size)

                # Сохранение статистики
                self.frame_sizes.append(frame_size)
                self.frame_count += 1

                # Отправка данных
                try:
                    self.sock.sendall(struct.pack(">L", frame_size) + data)
                except (BrokenPipeError, ConnectionResetError):
                    print("Соединение разорвано")
                    break

                # Контроль FPS
                processing_time = time.perf_counter() - frame_start
                wait_time = max(0, self.frame_interval - processing_time)
                time.sleep(wait_time)

        finally:
            self.running.clear()
            cap.release()
            self.sock.close()
            print("\n[ИТОГО] Поток остановлен")
            self.show_final_statistics()

    def show_final_statistics(self):
        """Отображение финальной статистики и графиков"""
        if not self.start_time or self.frame_count == 0:
            return

        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time
        total_data = sum(self.frame_sizes) * 8  # биты
        avg_bitrate = total_data / total_time

        print("\n" + "=" * 50)
        print(f"ФИНАЛЬНАЯ СТАТИСТИКА")
        print(f"Общее время: {total_time:.1f} сек")
        print(f"Всего кадров: {self.frame_count}")
        print(f"Средний FPS: {avg_fps:.1f}")
        print(f"Средний битрейт: {avg_bitrate / 1000000:.3f} Мбит/с")
        print(f"Целевой битрейт: {self.target_bitrate / 1000000:.3f} Мбит/с")
        print(
            f"Диапазон качества JPEG: {min(self.quality_history)}-{max(self.quality_history)}"
        )
        print("=" * 50)

        # Построение графиков
        plt.figure(figsize=(12, 10))

        # График FPS
        plt.subplot(3, 1, 1)
        plt.plot(self.fps_history, "b-", label="Фактический FPS")
        plt.axhline(y=self.target_fps, color="r", linestyle="--", label="Целевой FPS")
        plt.title("Производительность FPS")
        plt.ylabel("Кадров/сек")
        plt.grid(True)
        plt.legend()

        # График битрейта (Мбит/с)
        plt.subplot(3, 1, 2)
        plt.plot(
            [b / 1000000 for b in self.bitrate_history],
            "g-",
            label="Фактический битрейт",
        )
        plt.axhline(
            y=self.target_bitrate / 1000000,
            color="r",
            linestyle="--",
            label="Целевой битрейт",
        )
        plt.title("Производительность битрейта")
        plt.ylabel("Мбит/сек")
        plt.grid(True)
        plt.legend()

        # График качества
        plt.subplot(3, 1, 3)
        plt.plot(self.quality_history, "m-", label="Качество JPEG")
        plt.title("Динамика качества")
        plt.ylabel("Качество (1-100)")
        plt.xlabel("Время (сек)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("stream_stats.png")
        print("Графики сохранены в stream_stats.png")


if __name__ == "__main__":
    client = VideoClient(host="192.168.2.151")  # Замените на IP сервера
    try:
        client.connect()
        client.start_stream()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        client.running.clear()

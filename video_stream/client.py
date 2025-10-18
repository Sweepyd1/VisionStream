import socket
import cv2 as cv
import numpy as np
import struct
import time
from typing import Optional, Tuple

class VideoClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        target_width: int = 600,
        target_height: int = 480,
        target_fps: int = 20,
        fourcc: Optional[Tuple[str, str, str, str]] = ("M", "J", "P", "G"),
        use_advanced_processing: bool = False,
        target_compression_size: Optional[int] = None,
        initial_quality: int = 30,
        report_interval: Optional[float] = None,
    ):
        self.host = host
        self.port = port
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.fourcc = fourcc
        self.use_advanced_processing = use_advanced_processing
        self.target_compression_size = target_compression_size
        self.initial_quality = initial_quality
        self.report_interval = report_interval

    def _create_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((self.host, self.port))
        return sock

    def _setup_capture(self, source):
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {source}")
        if self.fourcc:
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv.CAP_PROP_FPS, self.target_fps)
        return cap

    def _process_frame_v1(self, frame):
        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            frame = cv.resize(frame, (self.target_width, self.target_height))
        gray = frame[:, :, 0]
        _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
        for cnt in contours:
            if cv.contourArea(cnt) > 100:
                cv.drawContours(contour_img, [cnt], -1, 255, 1)
        return contour_img, len(contours)

    def _process_frame_v2(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv.dilate(edges, kernel, iterations=1)
        smoothed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(smoothed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        contour_img = np.zeros((h, w), dtype=np.uint8)
        for contour in contours:
            epsilon = 0.005 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            cv.drawContours(contour_img, [approx], -1, 255, 1)
        # Фиксированный размер для сжатия
        if self.target_compression_size:
            contour_img = cv.resize(contour_img, (480, 270))
        return contour_img, len(contours)

    def _compress_and_send(self, img, sock):
        quality = self.initial_quality
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        step = 0
        while step < 5:
            _, jpeg = cv.imencode(".jpg", img, encode_param)
            if not self.target_compression_size or len(jpeg) <= self.target_compression_size:
                break
            quality = max(5, quality - 5)
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
            step += 1
        data = jpeg.tobytes()
        sock.sendall(struct.pack("Q", len(data)) + data)
        return len(data) + 8, quality

    def _print_stats_v1(self, bytes_sent, proc_time, avg_lat, fps, contours):
        print(
            f"Битрейт: {bytes_sent * 8 / 1e6:.2f} Мбит/с | "
            f"Задержка: {proc_time:.3f} сек | "
            f"Средняя: {avg_lat:.3f} сек | "
            f"FPS: {fps} | Контуров: {contours}"
        )

    def _print_stats_v2(self, total_bytes, elapsed, fps_actual, fps_target, size, quality, contours):
        bitrate = total_bytes * 8 / elapsed / 1e6
        print(
            f"Битрейт: {bitrate:.3f} Мбит/с | "
            f"FPS: {fps_actual:.1f}/{fps_target} | "
            f"Размер: {size} байт | Качество: {quality} | Контуры: {contours}"
        )

    def run(self, source: int | str = 0):
        sock = self._create_socket()
        cap = self._setup_capture(source)
        try:
            total_latency = 0.0
            frame_count = 0
            total_bytes = 0
            last_report = time.time()
            frame_delay = 1.0 / self.target_fps

            while True:
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                if self.use_advanced_processing:
                    img, cnt = self._process_frame_v2(frame)
                else:
                    img, cnt = self._process_frame_v1(frame)

                sent_bytes, quality = self._compress_and_send(img, sock)
                total_bytes += sent_bytes
                proc_time = time.time() - start

                frame_count += 1
                total_latency += proc_time

                if self.report_interval is None:
                    avg_lat = total_latency / frame_count
                    self._print_stats_v1(sent_bytes, proc_time, avg_lat, self.target_fps, cnt)
                    total_bytes = 0
                else:
                    now = time.time()
                    if now - last_report >= self.report_interval:
                        elapsed = now - last_report
                        actual_fps = frame_count / elapsed
                        self._print_stats_v2(total_bytes, elapsed, actual_fps, self.target_fps, sent_bytes, quality, cnt)
                        total_bytes = 0
                        frame_count = 0
                        last_report = now

                cv.imshow("Preview", frame)
                if cv.waitKey(1) == 27:
                    break

                if self.report_interval is not None:
                    sleep_time = max(0, frame_delay - proc_time)
                    time.sleep(sleep_time)

        finally:
            cap.release()
            cv.destroyAllWindows()
            sock.close()
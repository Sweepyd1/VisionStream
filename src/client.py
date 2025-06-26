import numpy as np
import cv2 as cv
import socket
import struct
import time
import zlib

host_ip = ""
port = 9999


def run_client():
    bytes_sent = 0

    target_fps = 20
    frame_delay = 1.0 / target_fps

    target_width, target_height = 600, 480

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.connect((host_ip, port))

    cap = cv.VideoCapture(0)

    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv.CAP_PROP_FPS, target_fps)

    contour_img = np.zeros((target_height, target_width), dtype=np.uint8)

    try:
        total_latency = 0
        frame_count = 0

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            gray = frame[:, :, 0]

            _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

            contours, _ = cv.findContours(
                binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            contour_img.fill(0)

            for cnt in contours:
                if cv.contourArea(cnt) > 100:
                    cv.drawContours(contour_img, [cnt], -1, 255, 1)

            # Используем быстрое сжатие JPEG с низким качеством
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 30]
            _, jpeg_frame = cv.imencode(".jpg", contour_img, encode_param)
            data = jpeg_frame.tobytes()

            message_size = struct.pack("Q", len(data))
            client_socket.sendall(message_size + data)
            bytes_sent += len(message_size) + len(data)

            processing_time = time.time() - frame_start

            if processing_time < frame_delay:
                target_fps = min(30, target_fps + 2)
            else:
                target_fps = max(15, target_fps - 5)

            frame_delay = 1.0 / target_fps

            frame_count += 1
            total_latency += processing_time
            avg_latency = total_latency / frame_count
            bitrate_mbps = bytes_sent * 8 / 1_000_000
            print(
                f"Битрейт: {bitrate_mbps:.2f} Мбит/с | "
                f"Задержка: {processing_time:.3f} сек | "
                f"Средняя: {avg_latency:.3f} сек | "
                f"FPS: {target_fps} | "
                f"Контуров: {len(contours)}"
            )
            bytes_sent = 0

            cv.imshow("Preview", contour_img)
            cv.waitKey(1)

    except (KeyboardInterrupt, socket.error):
        pass
    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()


def run_client_v2():
    total_bytes = 0
    last_report_time = time.time()
    frame_count = 0
    target_fps = 25
    frame_delay = 1.0 / target_fps

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.connect((host_ip, port))

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, target_fps)

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gaussian_kernel = 5
    contour_img = np.zeros((720, 1280), dtype=np.uint8)

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Гауссово размытие для уменьшения шума
            blurred = cv.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

            # Детектор границ Кэнни для получения непрерывных линий
            edges = cv.Canny(blurred, 50, 150)

            # Морфологические операции для сглаживания границ
            dilated = cv.dilate(edges, kernel, iterations=1)
            smoothed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

            # Поиск и аппроксимация контуров
            contours, _ = cv.findContours(
                smoothed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )

            contour_img.fill(0)
            for contour in contours:
                # Аппроксимация контура для сглаживания
                epsilon = 0.005 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True)
                cv.drawContours(contour_img, [approx], -1, 255, 1)

            small_contour = cv.resize(contour_img, (480, 270))

            quality = 10
            _, jpeg_frame = cv.imencode(
                ".jpg", small_contour, [int(cv.IMWRITE_JPEG_QUALITY), quality]
            )

            target_size = 1000
            step = 0
            while len(jpeg_frame) > target_size and step < 5:
                quality = max(5, quality - 5)
                _, jpeg_frame = cv.imencode(
                    ".jpg", small_contour, [int(cv.IMWRITE_JPEG_QUALITY), quality]
                )
                step += 1

            data = jpeg_frame.tobytes()
            frame_size = len(data)
            header = struct.pack("Q", frame_size)

            client_socket.sendall(header + data)
            total_bytes += len(header) + frame_size
            frame_count += 1

            current_time = time.time()
            elapsed = current_time - last_report_time
            if elapsed >= 1.0:
                bitrate = (total_bytes * 8) / elapsed / 1_000_000
                actual_fps = frame_count / elapsed
                print(
                    f"Битрейт: {bitrate:.3f} Мбит/с | "
                    f"FPS: {actual_fps:.1f}/{target_fps} | "
                    f"Размер: {frame_size} байт | "
                    f"Качество: {quality} | "
                    f"Контуры: {len(contours)}"
                )
                total_bytes = 0
                frame_count = 0
                last_report_time = current_time

            processing_time = current_time - start_time
            sleep_time = max(0, frame_delay - processing_time)
            time.sleep(sleep_time)

            cv.imshow("Preview", contour_img)
            if cv.waitKey(1) == 27:
                break

    except (KeyboardInterrupt, socket.error) as e:
        print(f"Ошибка: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()


total_bytes_sent = 0
frame_count = 0
start_time_stats = time.perf_counter()
target_fps = 15
frame_delay = 1.0 / target_fps


def print_stats(total_bytes, frame_count, start_time, processing_time, width, height):
    elapsed = time.perf_counter() - start_time
    if elapsed == 0:
        return
    fps = frame_count / elapsed
    bitrate_mbps = (total_bytes * 8) / (elapsed * 1_000_000)
    latency_ms = processing_time * 1000

    print(
        f"FPS: {fps:.2f} | "
        f"Битрейт: {bitrate_mbps:.4f} Мбит/с | "
        f"Разрешение: {width}x{height} | "
        f"Задержка: {latency_ms:.1f} мс"
    )


def serialize_contours(contours):
    scale_factor = 6
    min_points = 2

    data = bytearray()
    data.extend(struct.pack("H", len(contours)))

    for cnt in contours:
        epsilon = 0.005 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        points = (approx.reshape(-1, 2) / scale_factor).astype(np.int16)

        if len(points) < min_points:
            continue

        data.extend(struct.pack("H", len(points)))

        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                data.extend(struct.pack("hhhh", x1, y1, x2, y2))
            else:
                x, y = points[i]
                data.extend(struct.pack("hh", x, y))

    return zlib.compress(bytes(data), level=3)


def send_contour_data(socket, data):
    header = struct.pack("I", len(data))
    try:
        socket.sendall(header + data)
        return True
    except (BrokenPipeError, ConnectionResetError):
        print("Соединение разорвано")
        return False


def adaptive_fps_control(processing_time):
    global target_fps, frame_delay

    if processing_time < frame_delay * 0.8:
        target_fps = min(20, target_fps + 1)
    elif processing_time > frame_delay * 1.2:
        target_fps = max(8, target_fps - 2)

    frame_delay = 1.0 / target_fps


def run_client_v3():
    global total_bytes_sent, frame_count, start_time_stats, target_fps, frame_delay

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    try:
        client_socket.connect((host_ip, port))
        print("подключено")
    except ConnectionRefusedError:
        print("Не удалось подключиться к серверу")
        return

    target_width, target_height = 1280, 720
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv.CAP_PROP_FPS, 20)

    contour_min_area = 150
    gaussian_kernel = (3, 3)
    adaptive_block_size = 25

    while True:
        start_time = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(gray, gaussian_kernel, 0)

        binary = cv.adaptiveThreshold(
            blurred,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            adaptive_block_size,
            5,
        )

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        filtered_contours = [
            cnt for cnt in contours if cv.contourArea(cnt) > contour_min_area
        ]

        contour_data = serialize_contours(filtered_contours)

        if not send_contour_data(client_socket, contour_data):
            break

        processing_time = time.perf_counter() - start_time

        total_bytes_sent += len(contour_data) + 4
        frame_count += 1

        if frame_count % 10 == 0:
            print_stats(
                total_bytes_sent,
                frame_count,
                start_time_stats,
                processing_time,
                target_width,
                target_height,
            )

        adaptive_fps_control(processing_time)

        elapsed = time.perf_counter() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        try:
            cv.imshow("Preview", binary)
            if cv.waitKey(1) == ord("q"):
                break
        except Exception:
            pass

    cap.release()
    cv.destroyAllWindows()
    client_socket.close()
    print("Клиент остановлен")

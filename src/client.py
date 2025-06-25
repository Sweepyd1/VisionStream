import numpy as np
import cv2 as cv
import socket
import struct
import time
import zlib

host_ip = ""
port = 


def run_client():
    print(host_ip)
    print(port)
    bytes_sent = 0
    # Увеличиваем FPS для уменьшения задержки
    target_fps = 20
    frame_delay = 1.0 / target_fps

    # Уменьшаем разрешение для ускорения обработки
    target_width, target_height = 600, 480

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.connect((host_ip, port))

    cap = cv.VideoCapture(0)
    # Используем более быстрый видеопоток MJPEG
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv.CAP_PROP_FPS, target_fps)  # Устанавливаем FPS на уровне камеры

    # Оптимизация: предварительно выделяем память для часто используемых объектов
    contour_img = np.zeros((target_height, target_width), dtype=np.uint8)

    try:
        # Измеряем общую задержку
        total_latency = 0
        frame_count = 0

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 1. УСКОРЕНИЕ ОБРАБОТКИ ИЗОБРАЖЕНИЯ
            # Прямая работа с Y-каналом вместо конвертации в grayscale
            gray = frame[:, :, 0]  # Используем только синий канал (обычно самый чистый)

            # Быстрая бинаризация
            _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

            # Ускоренное нахождение контуров
            contours, _ = cv.findContours(
                binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            # Быстрая очистка изображения
            contour_img.fill(0)

            # 2. ОПТИМИЗАЦИЯ ОТРИСОВКИ КОНТУРОВ
            # Рисуем только крупные контуры
            for cnt in contours:
                if cv.contourArea(cnt) > 100:  # Порог площади контура
                    cv.drawContours(contour_img, [cnt], -1, 255, 1)

            # 3. УСКОРЕННОЕ СЖАТИЕ И ПЕРЕДАЧА
            # Используем быстрое сжатие JPEG с низким качеством
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 30]
            _, jpeg_frame = cv.imencode(".jpg", contour_img, encode_param)
            data = jpeg_frame.tobytes()

            # Отправляем без ожидания подтверждения
            message_size = struct.pack("Q", len(data))
            client_socket.sendall(message_size + data)
            bytes_sent += len(message_size) + len(data)
            # 4. УПРАВЛЕНИЕ ЧАСТОТОЙ КАДРОВ БЕЗ ОЖИДАНИЯ
            # Рассчитываем время обработки
            processing_time = time.time() - frame_start

            # Динамическая регулировка FPS
            if processing_time < frame_delay:
                # Увеличиваем FPS если успеваем обрабатывать
                target_fps = min(30, target_fps + 2)
            else:
                # Уменьшаем FPS если не успеваем
                target_fps = max(15, target_fps - 5)

            frame_delay = 1.0 / target_fps

            # 5. ВЫВОД ИНФОРМАЦИИ О ЗАДЕРЖКЕ
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

            # Быстрый показ превью
            cv.imshow("Preview", contour_img)
            cv.waitKey(1)  # Минимальное время ожидания

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

    # Параметры для сглаживания контуров
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

            # 1. Улучшенная обработка для гладких контуров
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

            # Отрисовка гладких контуров
            contour_img.fill(0)
            for contour in contours:
                # Аппроксимация контура для сглаживания
                epsilon = 0.005 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True)
                cv.drawContours(contour_img, [approx], -1, 255, 1)

            # 2. Оптимизация для снижения битрейта до 0.2 Мбит/с
            # Дополнительное уменьшение разрешения (1280x720 -> 480x270)
            small_contour = cv.resize(contour_img, (480, 270))

            # Более агрессивное сжатие
            quality = 10  # начальное качество
            _, jpeg_frame = cv.imencode(
                ".jpg", small_contour, [int(cv.IMWRITE_JPEG_QUALITY), quality]
            )

            # Регулировка качества для достижения целевого размера
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

            # Отправка данных
            client_socket.sendall(header + data)
            total_bytes += len(header) + frame_size
            frame_count += 1

            # Расчет и вывод статистики
            current_time = time.time()
            elapsed = current_time - last_report_time
            if elapsed >= 1.0:
                bitrate = (total_bytes * 8) / elapsed / 1_000_000  # Мбит/с
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

            # Регулировка FPS
            processing_time = current_time - start_time
            sleep_time = max(0, frame_delay - processing_time)
            time.sleep(sleep_time)

            # Быстрый просмотр
            cv.imshow("Preview", contour_img)
            if cv.waitKey(1) == 27:
                break

    except (KeyboardInterrupt, socket.error) as e:
        print(f"Ошибка: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()


# Глобальные переменные для статистики
total_bytes_sent = 0
frame_count = 0
start_time_stats = time.perf_counter()
target_fps = 15  # Начинаем с более низкого FPS
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
    """Сериализация контуров с улучшенной детализацией"""
    scale_factor = 6  # Уменьшаем квантование для большей точности
    min_points = 2  # Разрешаем контуры из 2 точек (линии)

    data = bytearray()
    data.extend(struct.pack("H", len(contours)))  # Количество контуров

    for cnt in contours:
        # Уменьшаем коэффициент упрощения для сохранения деталей
        epsilon = 0.005 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        # Квантование координат
        points = (approx.reshape(-1, 2) / scale_factor).astype(np.int16)

        # Разрешаем контуры с меньшим количеством точек
        if len(points) < min_points:
            continue

        data.extend(struct.pack("H", len(points)))  # Количество точек

        # Упаковка координат с группировкой
        for i in range(0, len(points), 2):  # Группируем по 2 точки
            if i + 1 < len(points):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                data.extend(struct.pack("hhhh", x1, y1, x2, y2))
            else:
                # Одиночная точка в конце
                x, y = points[i]
                data.extend(struct.pack("hh", x, y))

    # Применяем сжатие zlib с оптимальным уровнем
    return zlib.compress(bytes(data), level=3)


def send_contour_data(socket, data):
    """Отправка сжатых данных с заголовком"""
    header = struct.pack("I", len(data))
    try:
        socket.sendall(header + data)
        return True
    except (BrokenPipeError, ConnectionResetError):
        print("Соединение разорвано")
        return False


def adaptive_fps_control(processing_time):
    """Динамическая регулировка FPS с сохранением состояния"""
    global target_fps, frame_delay

    if processing_time < frame_delay * 0.8:
        target_fps = min(20, target_fps + 1)  # Медленное увеличение
    elif processing_time > frame_delay * 1.2:
        target_fps = max(8, target_fps - 2)  # Уменьшение при перегрузке

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

    # Уменьшаем детализацию для снижения битрейта
    contour_min_area = 150
    gaussian_kernel = (3, 3)
    adaptive_block_size = 25

    while True:
        start_time = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            break

        # 1. Быстрое преобразование в grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 2. Размытие для уменьшения детализации
        blurred = cv.GaussianBlur(gray, gaussian_kernel, 0)

        # 3. Адаптивная бинаризация с большим размером блока
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
        # 4. Нахождение только крупных контуров
        contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # 5. Фильтрация контуров по площади (увеличиваем минимальную площадь)
        filtered_contours = [
            cnt for cnt in contours if cv.contourArea(cnt) > contour_min_area
        ]

        # 6. Сериализация со сжатием
        contour_data = serialize_contours(filtered_contours)

        # 7. Отправка данных с проверкой соединения
        if not send_contour_data(client_socket, contour_data):
            break

        processing_time = time.perf_counter() - start_time

        # Обновление статистики
        total_bytes_sent += len(contour_data) + 4  # +4 для заголовка
        frame_count += 1

        # Печать статистики каждые 10 кадров
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

        # Управление задержкой для поддержания FPS
        elapsed = time.perf_counter() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Быстрый показ превью (если возможно)
        try:
            cv.imshow("Preview", binary)
            if cv.waitKey(1) == ord("q"):
                break
        except:
            # Игнорируем ошибки отображения
            pass

    # Закрытие ресурсов
    cap.release()
    cv.destroyAllWindows()
    client_socket.close()
    print("Клиент остановлен")

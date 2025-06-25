import socket
import time
import cv2 as cv
import zlib
import struct

import numpy as np
# from config import port

port = 


def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = "0.0.0.0"

    server_socket.bind((host_ip, port))
    server_socket.listen(1)
    print("сервер запущен!")

    client_socket, addr = server_socket.accept()
    client_socket.setsockopt(
        socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
    )  # Отключаем алгоритм Нейгла для уменьшения задержек
    client_socket.setsockopt(
        socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024
    )  # Устанавливаем размер буфера приема в 1 МБ

    data = b""
    payload_size = struct.calcsize("Q")

    try:
        while True:
            while len(data) < payload_size:
                packet = client_socket.recv(49152)
                if not packet:
                    print("Клиент отключился")
                    raise ConnectionResetError
                data += packet

            packed_msg_size = data[
                :payload_size
            ]  # Извлекаем первые 8 байт — размер сообщения
            data = data[payload_size:]  # Удаляем из буфера прочитанные байты
            msg_size = struct.unpack("Q", packed_msg_size)[
                0
            ]  # Распаковываем длину сообщения

            while len(data) < msg_size:
                packet = client_socket.recv(49152)
                if not packet:
                    print("Клиент отключился")
                    raise ConnectionResetError
                data += packet

            frame_data = data[:msg_size]  # Извлекаем байты кадра из буфера
            data = data[msg_size:]

            client_socket.sendall(
                b"ACK"
            )  # Отправляем клиенту подтверждение получения кадра

            try:
                decode_start = time.time()  # Запоминаем время начала декодирования
                # Декодируем JPEG-байты в изображение OpenCV
                frame = cv.imdecode(
                    np.frombuffer(frame_data, dtype=np.uint8), cv.IMREAD_COLOR
                )
                frame = cv.resize(frame, (1280, 720))
                decode_delay = time.time() - decode_start  # Время декодирования

                if frame.shape == 2:
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

                cv.imshow("Server Display", frame)
                print(f"Декодирование: {decode_delay:.4f} сек")
            except Exception as e:
                print(f"Ошибка декодирования: {e}")

            if cv.waitKey(1) == ord("q"):
                break

    except (ConnectionResetError, BrokenPipeError):
        print("Соединение закрыто клиентом")

    finally:
        client_socket.close()
        server_socket.close()
        cv.destroyAllWindows()


def receive_data(socket, size):
    """Надежное получение данных заданного размера"""
    data = bytearray()
    while len(data) < size:
        packet = socket.recv(min(4096, size - len(data)))
        if not packet:
            raise ConnectionError("Соединение потеряно")
        data.extend(packet)
    return bytes(data)


def deserialize_contours(data):
    """Десериализация контурных данных с учетом изменений в клиенте"""
    try:
        # Распаковываем данные
        decompressed = zlib.decompress(data)
    except zlib.error:
        print("Ошибка распаковки данных")
        return []

    contours = []
    offset = 0
    data_len = len(decompressed)

    # Проверяем минимальный размер данных
    if data_len < 2:
        return contours

    # Читаем количество контуров
    num_contours = struct.unpack("H", decompressed[offset : offset + 2])[0]
    offset += 2

    scale_factor = 8  # Должно совпадать с клиентом

    for _ in range(num_contours):
        # Проверяем доступность данных
        if offset + 2 > data_len:
            break

        # Читаем количество точек
        num_points = struct.unpack("H", decompressed[offset : offset + 2])[0]
        offset += 2

        # Проверяем доступность данных для точек
        if offset + num_points * 4 > data_len:
            break

        points = []
        for _ in range(num_points):
            # Читаем координаты (2 short)
            x, y = struct.unpack("hh", decompressed[offset : offset + 4])
            offset += 4

            # Масштабируем координаты обратно
            x = int(x * scale_factor)
            y = int(y * scale_factor)
            points.append([x, y])

        # Преобразуем в формат OpenCV
        if len(points) > 0:
            contours.append(np.array(points, dtype=np.int32).reshape(-1, 1, 2))

    return contours


def run_server_v2():
    host_ip = "0.0.0.0"
    port = 9999

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host_ip, port))
    server_socket.listen(1)
    print("сервер запущен!")

    client_socket, addr = server_socket.accept()
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"Подключен клиент: {addr}")

    try:
        while True:
            # 1. Получаем размер данных (4 байта)
            try:
                header = receive_data(client_socket, 4)
                data_size = struct.unpack("I", header)[0]
            except (struct.error, ConnectionError):
                print("Ошибка чтения заголовка")
                break

            # 2. Получаем данные контуров
            try:
                contour_data = receive_data(client_socket, data_size)
            except ConnectionError:
                print("Ошибка чтения данных контуров")
                break

            # 3. Десериализация контуров
            contours = deserialize_contours(contour_data)

            # 4. Визуализация
            display_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            if contours:
                cv.drawContours(display_image, contours, -1, (255, 255, 255), 2)

            # 5. Отображение
            cv.imshow("High-Resolution Contours", display_image)

            # 6. Отправляем подтверждение клиенту
            try:
                client_socket.sendall(b"ACK")
            except (BrokenPipeError, ConnectionResetError):
                print("Соединение с клиентом потеряно")
                break

            if cv.waitKey(1) == ord("q"):
                break

    except KeyboardInterrupt:
        print("Сервер остановлен по запросу пользователя")
    finally:
        client_socket.close()
        server_socket.close()
        cv.destroyAllWindows()
        print("Сервер остановлен")

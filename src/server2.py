import socket
import numpy as np
import cv2
import struct
from threading import Thread

class VideoServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print(f"[*] Сервер запущен на {host}:{port}")

        # Параметры потока
        self.target_bitrate = 100000  # 0.1 Мбит/с (в битах)
        self.target_fps = 25
        self.width, self.height = 1280, 720  # HD разрешение

    def start(self):
        conn, addr = self.sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[+] Подключено: {addr}")
        self.handle_client(conn)

    def handle_client(self, conn):
        payload_size = struct.calcsize(">L")
        data = b""

        while True:
            try:
                # Получаем размер кадра
                while len(data) < payload_size:
                    packet = conn.recv(4096)
                    if not packet: # Если клиент отключился
                        print("[-] Клиент отключился.")
                        break
                    data += packet
                if not packet: # Повторная проверка, если цикл прервался из-за отключения
                    break

                packed_size = data[:payload_size]
                data = data[payload_size:]
                size = struct.unpack(">L", packed_size)[0]

                # Получаем данные кадра
                while len(data) < size:
                    packet = conn.recv(4096)
                    if not packet: # Если клиент отключился
                        print("[-] Клиент отключился.")
                        break
                    data += packet
                if not packet: # Повторная проверка, если цикл прервался из-за отключения
                    break

                frame_data = data[:size]
                data = data[size:]

                # Декодируем кадр
                frame = cv2.imdecode(
                    np.frombuffer(frame_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is None: # Проверка, если декодирование не удалось
                    print("[!] Ошибка декодирования кадра. Пропускаем.")
                    continue

                # --- НОВАЯ ОБРАБОТКА: ОБНАРУЖЕНИЕ КОНТУРОВ ---
                # 1. Преобразовать в оттенки серого
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 2. Размыть для уменьшения шумов (необязательно, но улучшает результат)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # 3. Обнаружить контуры с помощью Canny
                # Пороговые значения можно настроить
                edges = cv2.Canny(blurred, 50, 150) # 50 и 150 - это нижний и верхний пороги

                # 4. (Опционально) Преобразовать контуры обратно в цветное изображение
                # для отображения на черном фоне или для более приятного вида
                # Создаем черное изображение и накладываем на него контуры белым цветом
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


                # Отображаем кадр с контурами
                cv2.imshow("Video Stream (Contours)", edges_colored)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except struct.error as e:
                print(f"[!] Ошибка struct.unpack: {e}. Возможно, неполные данные кадра. Пропускаем.")
                data = b"" # Очищаем буфер, чтобы начать получать следующий кадр
            except cv2.error as e:
                print(f"[!] Ошибка OpenCV: {e}. Пропускаем кадр.")
                data = b"" # Очищаем буфер
            except Exception as e:
                print(f"[!] Произошла непредвиденная ошибка: {e}. Закрытие соединения.")
                break # Выход из цикла при любой другой ошибке


        conn.close()
        cv2.destroyAllWindows()
        print("[-] Соединение закрыто.")


if __name__ == "__main__":
    server = VideoServer()
    server.start()

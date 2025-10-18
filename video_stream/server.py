import socket
import cv2 as cv
import numpy as np
import struct
from typing import Tuple

class VideoServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9999, display_size: Tuple[int, int] = (1280, 720)):
        self.host = host
        self.port = port
        self.display_size = display_size

    def run(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        print(f"Сервер запущен на {self.host}:{self.port}")

        try:
            client_sock, addr = server_sock.accept()
            print(f"Подключён клиент: {addr}")
            client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

            data = b""
            payload_size = struct.calcsize("Q")

            while True:
                while len(data) < payload_size:
                    packet = client_sock.recv(49152)
                    if not packet:
                        raise ConnectionResetError
                    data += packet

                msg_size = struct.unpack("Q", data[:payload_size])[0]
                data = data[payload_size:]

                while len(data) < msg_size:
                    packet = client_sock.recv(49152)
                    if not packet:
                        raise ConnectionResetError
                    data += packet

                frame_data = data[:msg_size]
                data = data[msg_size:]

                client_sock.sendall(b"ACK")

                try:
                    frame = cv.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv.IMREAD_COLOR)
                    if frame is None:
                        continue
                    frame = cv.resize(frame, self.display_size)
                    if len(frame.shape) == 2:
                        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                    cv.imshow("Server Display", frame)
                    if cv.waitKey(1) == ord("q"):
                        break
                except Exception as e:
                    print(f"Ошибка декодирования: {e}")

        except (ConnectionResetError, BrokenPipeError):
            print("Клиент отключился")
        finally:
            client_sock.close()
            server_sock.close()
            cv.destroyAllWindows()
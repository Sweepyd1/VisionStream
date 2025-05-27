import socket
import cv2
import pickle
import struct
from config import port
port = int(port)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '0.0.0.0'

server_socket.bind((host_ip, port))
server_socket.listen(5)

client_socket, addr = server_socket.accept()
data = b""
payload_size = struct.calcsize("Q")

try:
    while True:
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4096)
        
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        try:
            frame = pickle.loads(frame_data)
            if len(frame.shape) == 2:  
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            cv2.imshow("Server Display", frame)
        except Exception as e:
            print(f"Ошибка декодирования: {e}")

        if cv2.waitKey(1) == ord('q'): 
            break

finally:
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
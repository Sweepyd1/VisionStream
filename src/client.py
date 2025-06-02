import numpy as np
import cv2 as cv
import socket

import struct
from config import host_ip, port
import time
port = int(port)

def run_client():
    frame_rate = 25  
    frame_delay = 1.0 / frame_rate

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    client_socket.connect((host_ip, port))

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv.resize(frame, (1280, 720))
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray, 100, 200)
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contour_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            cv.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 30]  
            _, jpeg_frame = cv.imencode('.jpg', contour_img, encode_param)
            data = jpeg_frame.tobytes()

            message_size = struct.pack("Q", len(data))
            client_socket.sendall(message_size + data)
            
            send_time = time.time() - start_time

            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

            ack = client_socket.recv(1024)

            rtt = time.time() - start_time
            
            data_size_bits = len(data) * 8  
            bitrate_mbps = (data_size_bits / send_time) / 1_000_000  

            print(f"Битрейт: {bitrate_mbps/1000:.2f} Мбит/с | Задержка (RTT): {rtt:.4f} сек")
            

            cv.imshow("Client Preview", contour_img)
 
            
            if cv.waitKey(1) == ord("q"):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()

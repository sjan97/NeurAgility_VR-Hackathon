# save as button_server.py
import socket
import time
import random

HOST = '127.0.0.1'  # Unity runs on same machine
PORT = 65432        # any free port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print("Waiting for Unity connection...")
conn, addr = s.accept()
print(f"Connected by {addr}")

try:
    while True:
        # Generate a random position for the button
        x = random.uniform(-5, 5)
        y = random.uniform(-3, 3)
        z = 0  # For 2D button in Canvas you can ignore z

        # Send as a string
        msg = f"{x},{y},{z}\n"
        conn.sendall(msg.encode('utf-8'))
        time.sleep(0.05)  # 20 FPS
finally:
    conn.close()

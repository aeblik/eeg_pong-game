# pong_udp.py
# Simple Pong controlled by UDP "EVENT:BLINK" messages.
# Tweaks:
#  - Slower overall speed
#  - No gravity
#  - Paddle moves at constant speed; blink toggles direction.

import pygame
import socket
import threading

W, H = 800, 480
PADDLE_W, PADDLE_H = 12, 80
BALL_SIZE = 12
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

UDP_HOST, UDP_PORT = "127.0.0.1", 8765

class UdpListener(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.last_event = None

    def run(self):
        while True:
            data, _ = self.sock.recvfrom(1024)
            self.last_event = data.decode("utf-8").strip()

    def poll(self):
        ev = self.last_event
        self.last_event = None
        return ev

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("EEG Pong (Blink-controlled)")
    clock = pygame.time.Clock()

    # --- Speeds (tuned slower) ---
    paddle_speed = 140.0   # constant speed in pixels/second
    ball_speed_x = 160.0   # slower ball than before
    ball_speed_y = 120.0

    # Paddle
    paddle_y = H // 2 - PADDLE_H // 2
    paddle_dir = 0          # -1 up, +1 down, 0 stopped
    paddle_v = 0.0

    # Ball
    ball_x, ball_y = W // 2, H // 2
    ball_vx, ball_vy = ball_speed_x, ball_speed_y

    # UDP listener
    listener = UdpListener()
    listener.start()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # seconds

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # --- Read EEG events ---
        evt = listener.poll()
        if evt == "EVENT:BLINK":
            # toggle direction: if stopped, start going up
            if paddle_dir == 0:
                paddle_dir = -1  # start moving up
            else:
                paddle_dir *= -1  # flip between up/down

        # Update paddle velocity from direction
        paddle_v = paddle_dir * paddle_speed

        # Update paddle position
        paddle_y += paddle_v * dt

        # Clamp paddle inside screen; if hits edge, flip direction so it bounces
        if paddle_y <= 0:
            paddle_y = 0
            if paddle_dir < 0:
                paddle_dir = 1
        elif paddle_y >= H - PADDLE_H:
            paddle_y = H - PADDLE_H
            if paddle_dir > 0:
                paddle_dir = -1

        # --- Update ball ---
        ball_x += ball_vx * dt
        ball_y += ball_vy * dt

        # Top/bottom walls
        if ball_y <= 0:
            ball_y = 0
            ball_vy = abs(ball_vy)
        elif ball_y >= H - BALL_SIZE:
            ball_y = H - BALL_SIZE
            ball_vy = -abs(ball_vy)

        # Paddle collision (left)
        if (ball_x <= 40 + PADDLE_W and
            ball_x >= 40 and
            paddle_y <= ball_y <= paddle_y + PADDLE_H):
            ball_x = 40 + PADDLE_W
            ball_vx = abs(ball_vx)

        # Right wall
        if ball_x >= W - BALL_SIZE:
            ball_x = W - BALL_SIZE
            ball_vx = -abs(ball_vx)

        # Missed ball (left wall): reset
        if ball_x < 0:
            ball_x, ball_y = W // 2, H // 2
            ball_vx = ball_speed_x
            ball_vy = ball_speed_y

        # --- Draw ---
        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, (40, paddle_y, PADDLE_W, PADDLE_H))
        pygame.draw.rect(screen, WHITE, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

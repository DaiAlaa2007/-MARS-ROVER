"""
Mars Rover Pathfinder - Member 3
Rover Simulation & Mars Environment

Files required in the same folder:
    mars_rover_astar.py
Optional rock classification:
    rock_detector.py
    rock_model.pkl

Run:
    pip install pygame opencv-python numpy matplotlib
    python rover_simulation.py

Controls:
    SPACE : pause / resume rover
    O     : place a new obstacle ahead of the rover and re-plan
    C     : capture a simulated camera frame and try rock classification
    R     : restart the mission
    Mouse : click an empty map cell to add an obstacle
    ESC   : quit
"""

import os
import sys
import math
import pygame
import numpy as np
import cv2

from mars_rover_astar import (
    MarsMap, AStarPlanner, TERRAIN,
    FLAT, SAND, ROCK, SLOPE, STORM, OBSTACLE
)

# Rock detector is optional. It needs rock_model.pkl in the same folder.
try:
    from rock_detector import analyze_rock
    ROCK_MODEL_AVAILABLE = True
except Exception as exc:
    analyze_rock = None
    ROCK_MODEL_AVAILABLE = False
    ROCK_MODEL_ERROR = str(exc)


# --------------------------- Window settings ---------------------------

CELL = 22
MAP_COLS = 40
MAP_ROWS = 30
MAP_X = 20
MAP_Y = 48
PANEL_X = MAP_X + MAP_COLS * CELL + 24
PANEL_W = 300
WIDTH = PANEL_X + PANEL_W + 20
HEIGHT = MAP_Y + MAP_ROWS * CELL + 30
FPS = 60
MOVE_INTERVAL = 0.16

BG = (18, 22, 30)
PANEL = (29, 36, 48)
WHITE = (240, 242, 245)
MUTED = (170, 180, 192)
GREEN = (40, 235, 80)
RED = (245, 65, 65)
BLUE = (45, 155, 245)
YELLOW = (255, 210, 50)
PURPLE = (180, 120, 235)


def cell_center(row, col):
    return (
        MAP_X + col * CELL + CELL // 2,
        MAP_Y + row * CELL + CELL // 2
    )


class MarsRoverSimulation:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Mars Rover Pathfinder - Member 3 Simulation")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.small = pygame.font.SysFont("arial", 14)
        self.big = pygame.font.SysFont("arial", 25, bold=True)

        self.planner = AStarPlanner()
        self.reset()

    def reset(self):
        self.mars = MarsMap.generate(rows=MAP_ROWS, cols=MAP_COLS, seed=7)
        self.plan = self.planner.plan(self.mars)
        self.path = list(self.plan.path)
        self.path_index = 0
        self.rover_pos = self.path[0] if self.path else self.mars.start
        self.moving = True
        self.last_move = 0
        self.discovered = []
        self.plan_count = 0
        self.message = "Mission started. Rover is moving to the target."
        self.camera_surface = None
        self.scan_result = "No scan yet."
        self.energy_left = 100.0
        self.mission_complete = False

    def draw_text(self, text, x, y, color=WHITE, font=None):
        font = font or self.font
        surface = font.render(str(text), True, color)
        self.screen.blit(surface, (x, y))

    def draw_map(self):
        # Draw terrain cells.
        for r in range(self.mars.rows):
            for c in range(self.mars.cols):
                kind = int(self.mars.grid[r, c])
                color = tuple(int(TERRAIN[kind][3].lstrip("#")[i:i+2], 16)
                              for i in (0, 2, 4))
                rect = pygame.Rect(MAP_X + c * CELL, MAP_Y + r * CELL,
                                   CELL, CELL)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Planned route.
        if len(self.path) > 1:
            points = [cell_center(r, c) for r, c in self.path]
            pygame.draw.lines(self.screen, BLUE, False, points, 4)

        # Discovered obstacles.
        for r, c in self.discovered:
            x, y = cell_center(r, c)
            pygame.draw.line(self.screen, RED, (x - 7, y - 7), (x + 7, y + 7), 4)
            pygame.draw.line(self.screen, RED, (x - 7, y + 7), (x + 7, y - 7), 4)

        # Start and goal markers.
        sx, sy = cell_center(*self.mars.start)
        gx, gy = cell_center(*self.mars.goal)
        pygame.draw.circle(self.screen, GREEN, (sx, sy), 9)
        pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), 9, 2)
        pygame.draw.polygon(self.screen, YELLOW,
                            [(gx, gy - 12), (gx + 10, gy + 9),
                             (gx - 10, gy + 9)])
        pygame.draw.polygon(self.screen, (0, 0, 0),
                            [(gx, gy - 12), (gx + 10, gy + 9),
                             (gx - 10, gy + 9)], 2)

        # Rover body and direction marker.
        rr, rc = self.rover_pos
        rx, ry = cell_center(rr, rc)
        body = pygame.Rect(rx - 8, ry - 7, 16, 14)
        pygame.draw.rect(self.screen, (205, 215, 225), body, border_radius=4)
        pygame.draw.rect(self.screen, (35, 45, 58), body, 2, border_radius=4)
        pygame.draw.circle(self.screen, (35, 45, 58), (rx - 5, ry - 8), 3)
        pygame.draw.circle(self.screen, (35, 45, 58), (rx + 5, ry - 8), 3)
        pygame.draw.circle(self.screen, (35, 45, 58), (rx - 5, ry + 8), 3)
        pygame.draw.circle(self.screen, (35, 45, 58), (rx + 5, ry + 8), 3)

    def draw_panel(self):
        panel_rect = pygame.Rect(PANEL_X, 15, PANEL_W, HEIGHT - 30)
        pygame.draw.rect(self.screen, PANEL, panel_rect, border_radius=12)

        x = PANEL_X + 16
        y = 30
        self.draw_text("MISSION CONTROL", x, y, YELLOW, self.big)
        y += 45
        self.draw_text("ROVER STATUS", x, y, BLUE)
        y += 28
        self.draw_text(f"Position: {self.rover_pos}", x, y)
        y += 25
        self.draw_text(f"Destination: {self.mars.goal}", x, y)
        y += 25
        status = "COMPLETED" if self.mission_complete else (
            "PAUSED" if not self.moving else "MOVING")
        self.draw_text(f"Status: {status}",
                       x, y, GREEN if self.mission_complete else WHITE)
        y += 34

        self.draw_text("NAVIGATION", x, y, BLUE)
        y += 27
        self.draw_text(f"Path length: {len(self.path)} cells", x, y, MUTED, self.small)
        y += 21
        self.draw_text(f"Re-plans: {self.plan_count}", x, y, MUTED, self.small)
        y += 21
        if self.plan and self.plan.found:
            self.draw_text(f"Distance: {self.plan.distance:.1f}", x, y, MUTED, self.small)
            y += 21
            self.draw_text(f"Energy cost: {self.plan.energy:.1f}", x, y, MUTED, self.small)
            y += 21
            self.draw_text(f"Terrain risk: {self.plan.risk:.1f}", x, y, MUTED, self.small)
            y += 21
            self.draw_text(f"Total cost: {self.plan.total_cost:.1f}", x, y, MUTED, self.small)
            y += 21
        y += 12

        self.draw_text("BATTERY (estimated)", x, y, BLUE)
        y += 25
        bar = pygame.Rect(x, y, PANEL_W - 32, 15)
        pygame.draw.rect(self.screen, (70, 75, 85), bar, border_radius=5)
        fill = bar.copy()
        fill.width = int(bar.width * max(0, self.energy_left) / 100)
        pygame.draw.rect(self.screen, GREEN if self.energy_left > 25 else RED,
                         fill, border_radius=5)
        y += 30
        self.draw_text(f"{max(0, self.energy_left):.1f}%", x, y, MUTED, self.small)
        y += 34

        self.draw_text("CAMERA / ROCK SCAN", x, y, BLUE)
        y += 25
        if self.camera_surface is not None:
            preview = pygame.transform.smoothscale(self.camera_surface, (PANEL_W - 32, 105))
            self.screen.blit(preview, (x, y))
            y += 112
        else:
            self.draw_text("Press C to capture camera frame", x, y, MUTED, self.small)
            y += 23
        for line in self.scan_result.split("\n")[:3]:
            self.draw_text(line[:34], x, y, MUTED, self.small)
            y += 19

        y = HEIGHT - 170
        self.draw_text("CONTROLS", x, y, YELLOW)
        y += 25
        for line in ["SPACE  Pause / Resume",
                     "O      Add obstacle ahead",
                     "C      Camera scan",
                     "R      Restart mission",
                     "CLICK  Add obstacle on map",
                     "ESC    Quit"]:
            self.draw_text(line, x, y, WHITE, self.small)
            y += 20

    def add_obstacle(self, cell):
        r, c = cell
        if not self.mars.in_bounds(r, c):
            return
        if cell in (self.mars.start, self.mars.goal, self.rover_pos):
            self.message = "Cannot block the start, goal, or current rover cell."
            return
        if self.mars.is_blocked(r, c):
            self.message = "That cell is already blocked."
            return

        self.mars.add_obstacle(r, c)
        self.discovered.append(cell)
        new_plan = self.planner.plan(self.mars, start=self.rover_pos,
                                     goal=self.mars.goal)
        self.plan_count += 1

        if new_plan.found:
            self.plan = new_plan
            self.path = list(new_plan.path)
            self.path_index = 0
            self.message = f"New obstacle at {cell}. A* calculated a new route."
            self.moving = True
        else:
            self.moving = False
            self.message = "Warning: no safe path to the target was found."

    def add_obstacle_ahead(self):
        # Put an obstacle several steps ahead on the current route.
        if len(self.path) <= self.path_index + 3:
            self.message = "There is no route segment ahead to block."
            return
        target_index = min(self.path_index + 5, len(self.path) - 2)
        self.add_obstacle(self.path[target_index])

    def capture_camera(self):
        # Simulated rover camera: capture a region around the rover on the map.
        rr, rc = self.rover_pos
        radius = 4
        r1, r2 = max(0, rr - radius), min(self.mars.rows, rr + radius + 1)
        c1, c2 = max(0, rc - radius), min(self.mars.cols, rc + radius + 1)

        camera = pygame.Surface(((c2 - c1) * CELL, (r2 - r1) * CELL))
        camera.fill((180, 100, 55))
        for r in range(r1, r2):
            for c in range(c1, c2):
                kind = int(self.mars.grid[r, c])
                hex_color = TERRAIN[kind][3].lstrip("#")
                color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                rect = pygame.Rect((c - c1) * CELL, (r - r1) * CELL, CELL, CELL)
                pygame.draw.rect(camera, color, rect)
        # Draw a sample rock in the camera image to represent the camera view.
        cx, cy = camera.get_width() // 2, camera.get_height() // 2
        pygame.draw.ellipse(camera, (75, 68, 60),
                            (cx - 17, cy - 12, 34, 24))
        pygame.draw.ellipse(camera, (125, 112, 95),
                            (cx - 11, cy - 8, 12, 7))
        self.camera_surface = camera

        os.makedirs("captures", exist_ok=True)
        capture_path = os.path.join("captures", "rover_camera.png")
        pygame.image.save(camera, capture_path)

        if ROCK_MODEL_AVAILABLE:
            rgb = pygame.surfarray.array3d(camera)
            rgb = np.transpose(rgb, (1, 0, 2))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            try:
                result = analyze_rock(bgr)
                self.scan_result = (
                    f"Rock: {result['rock_type']}\n"
                    f"Mineral: {result['possible_mineral']}\n"
                    f"Confidence: {result['confidence']}%"
                )
            except Exception as exc:
                self.scan_result = f"Classifier error:\n{str(exc)[:45]}"
        else:
            self.scan_result = (
                "Camera frame captured.\n"
                "Classifier unavailable:\n"
                "Add rock_model.pkl to enable AI."
            )
        self.message = f"Camera frame saved: {capture_path}"

    def update_rover(self):
        if not self.moving or self.mission_complete or not self.path:
            return
        now = pygame.time.get_ticks() / 1000.0
        if now - self.last_move < MOVE_INTERVAL:
            return
        self.last_move = now

        if self.path_index + 1 < len(self.path):
            self.path_index += 1
            self.rover_pos = self.path[self.path_index]
            self.energy_left = max(0, self.energy_left - 0.12)
        else:
            if self.rover_pos == self.mars.goal:
                self.mission_complete = True
                self.moving = False
                self.message = "Mission complete! Rover reached the rock sample."
            else:
                self.moving = False
                self.message = "Rover stopped: route ended before the target."

    def handle_click(self, pos):
        mx, my = pos
        if MAP_X <= mx < MAP_X + MAP_COLS * CELL and MAP_Y <= my < MAP_Y + MAP_ROWS * CELL:
            c = (mx - MAP_X) // CELL
            r = (my - MAP_Y) // CELL
            self.add_obstacle((int(r), int(c)))

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not self.mission_complete:
                            self.moving = not self.moving
                            self.message = "Rover resumed." if self.moving else "Rover paused."
                    elif event.key == pygame.K_o:
                        self.add_obstacle_ahead()
                    elif event.key == pygame.K_c:
                        self.capture_camera()
                    elif event.key == pygame.K_r:
                        self.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            self.update_rover()
            self.screen.fill(BG)
            self.draw_text("MARS ROVER PATHFINDER", MAP_X, 12, WHITE, self.big)
            self.draw_map()
            self.draw_panel()

            # Message strip under the map.
            msg_y = MAP_Y + MAP_ROWS * CELL + 5
            self.draw_text(self.message[:105], MAP_X, msg_y, YELLOW, self.small)

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    try:
        MarsRoverSimulation().run()
    except ImportError as exc:
        print("Missing library:", exc)
        print("Install with: pip install pygame opencv-python numpy matplotlib")
        sys.exit(1)

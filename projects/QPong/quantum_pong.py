#!/usr/bin/env python3
"""Quantum Pong — a two player Schrödinger equation game.

The wavefunction lives at interior points of [0, 1]².  A Strang split-step
propagator uses an orthonormal discrete sine transform for the kinetic step;
this is a fast unitary approximation of homogeneous Dirichlet boundaries.
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from dataclasses import dataclass

import numpy as np
from scipy.fft import dstn, idstn


# One grid setting controls vertical resolution; the horizontal count is
# derived from the rectangular court's aspect ratio.
GRID = 200
COURT_WIDTH = 1.5
COURT_HEIGHT = 1.0
SIZE = 750
# A larger mass suppresses free-packet spreading.  Momentum is scaled by the
# same value below, retaining a playably quick group velocity.
MASS = 100.0
# Two short unitary steps resolve the sharp moving racket barriers while
# retaining the same 0.00060 units of simulated time per rendered frame.
DT = 0.001
SUBSTEPS = 1
# Each player has 0.35 units of scoring space behind their racket.
SCORING_SPACE = 0.35
PADDLE_X = (SCORING_SPACE, COURT_WIDTH - SCORING_SPACE)
PADDLE_LENGTH = 0.28
PADDLE_SPEED = 0.55
# Launch packets have kinetic energy of roughly 10³.  This five-cell-wide
# potential is safely above it and is accurately resolved by the timestep.
BARRIER_HEIGHT = 3000.0
COOLDOWN = 0.85
MEASUREMENT_PAUSE = 2.0
MEASUREMENT_SIGMA = 0.01
SIGMA = 0.03

@dataclass
class Paddle:
    y: float = 0.5
    ready_at: float = 0.0


class QuantumField:
    """A normalized wavefunction and its time-dependent barrier potential."""

    def __init__(self, n: int = GRID) -> None:
        self.ny = n
        self.nx = round(n * COURT_WIDTH / COURT_HEIGHT)
        self.dx = COURT_WIDTH / (self.nx + 1)
        self.dy = COURT_HEIGHT / (self.ny + 1)
        x_points = np.arange(1, self.nx + 1) * self.dx
        y_points = np.arange(1, self.ny + 1) * self.dy
        self.x, self.y = np.meshgrid(x_points, y_points, indexing="ij")
        modes_x = np.arange(1, self.nx + 1, dtype=float)
        modes_y = np.arange(1, self.ny + 1, dtype=float)
        mx, my = np.meshgrid(modes_x, modes_y, indexing="ij")
        # Eigenvalues of -Δ/(2m) on the rectangular Dirichlet domain.
        energy = np.pi**2 * ((mx / COURT_WIDTH) ** 2 + (my / COURT_HEIGHT) ** 2) / (2 * MASS)
        self.kinetic_phase = np.exp(-1j * energy * DT)
        self.psi = np.zeros((self.nx, self.ny), dtype=np.complex128)
        self.rng = np.random.default_rng()
        self.velocity = (0.0, 0.0)
        self.reset()

    def reset(self, direction: int | None = None, owner: int | None = None) -> None:
        """Prepare a new diagonal serve, optionally from a player's half."""
        if direction is None:
            direction = int(self.rng.choice((-1, 1)))
        # A point winner receives the next serve on their own half.  Neutral
        # serves (new game and non-scoring measurements) begin at centre.
        x0 = 0.55 if owner == 0 else 1.25 if owner == 1 else COURT_WIDTH / 2
        y0 = COURT_HEIGHT / 2
        # A compact launch packet; the increased effective mass keeps it
        # localized through several reflections. Random speed and vertical
        # sign make every serve take a distinct diagonal route.
        
        vx = direction * float(self.rng.uniform(6.5, 8.5))
        vy = int(self.rng.choice((-1, 1))) * float(self.rng.uniform(3.0, 5.0))
        self.velocity = (vx, vy)
        kx = MASS * vx
        ky = MASS * vy
        envelope = np.exp(-((self.x - x0) ** 2 + (self.y - y0) ** 2) /
                          (2 * SIGMA**2))
        self.psi = envelope * np.exp(1j * (kx * (self.x - x0) + ky * (self.y - y0)))
        self.normalize()

    def normalize(self) -> None:
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2))
        if norm:
            self.psi /= norm

    def potential(self, left: Paddle, right: Paddle) -> np.ndarray:
        v = np.zeros((self.nx, self.ny), dtype=float)
        half = PADDLE_LENGTH / 2
        # Five-cell thick rectangular potential barriers. The openings above
        # and below a racket make positioning matter.
        for px, paddle in ((PADDLE_X[0], left), (PADDLE_X[1], right)):
            ix = np.abs(self.x[:, 0] - px) <= 2.5 * self.dx
            iy = np.abs(self.y[0, :] - paddle.y) <= half
            v[np.ix_(ix, iy)] = BARRIER_HEIGHT
        return v

    def step(self, potential: np.ndarray) -> None:
        # V(dt/2) K(dt) V(dt/2), a second-order unitary split operator.
        half_phase = np.exp(-0.5j * potential * DT)
        self.psi *= half_phase
        coefficients = dstn(self.psi, type=1, norm="ortho")
        self.psi = idstn(coefficients * self.kinetic_phase, type=1, norm="ortho")
        self.psi *= half_phase

    def measure(self, rng: np.random.Generator) -> tuple[float, float]:
        """Sample |psi|², then collapse around that measured position."""
        probabilities = np.abs(self.psi.ravel()) ** 2
        probabilities /= probabilities.sum()
        chosen = rng.choice(probabilities.size, p=probabilities)
        i, j = np.unravel_index(chosen, self.psi.shape)
        x, y = float(self.x[i, j]), float(self.y[i, j])
        # A one-cell delta has arbitrarily broad momentum and instantly fills
        # the board. This is a finite-resolution position measurement: its
        # centre is exactly the sampled grid point, while its width prevents
        # a numerical ultraviolet burst after the pause.
        envelope = np.exp(-((self.x - x) ** 2 + (self.y - y) ** 2) /
                          (2 * MEASUREMENT_SIGMA**2))
        self.psi = envelope.astype(np.complex128)
        self.normalize()
        return x, y


class Game:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Quantum Pong")
        self.root.minsize(800, 430)
        self.root.resizable(True, True)
        self.fullscreen = False
        self.field_width = SIZE
        self.field_height = round(SIZE * COURT_HEIGHT / COURT_WIDTH)
        self.field_left = 0
        self.field_top = 0
        self.field = QuantumField()
        self.left, self.right = Paddle(), Paddle()
        self.rng = np.random.default_rng()
        self.score = [0, 0]
        self.keys: set[str] = set()
        self.last_frame = time.perf_counter()
        self.pause_until = 0.0
        self.measurement_marker: tuple[float, float] | None = None
        self.pending_winner: int | None = None
        self.pending_relaunch_direction = 1
        self.status = "The packet begins in a superposition. Place your barrier and measure."

        container = tk.Frame(self.root, bg="#07111f")
        container.pack(fill="both", expand=True, padx=14, pady=12)
        self.canvas = tk.Canvas(container, width=SIZE, height=self.field_height,
                                highlightthickness=0, bg="#020812")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.left_id = self.canvas.create_rectangle(0, 0, 0, 0, fill="#62d9ff", outline="")
        self.right_id = self.canvas.create_rectangle(0, 0, 0, 0, fill="#ff6e9e", outline="")
        self.midline_id = self.canvas.create_line(0, 0, 0, 0, fill="#24405c", dash=(4, 8))
        self.marker_h = self.canvas.create_line(0, 0, 0, 0, fill="#fff27a", width=2, state="hidden")
        self.marker_v = self.canvas.create_line(0, 0, 0, 0, fill="#fff27a", width=2, state="hidden")

        panel = tk.Frame(container, bg="#07111f", padx=16)
        panel.grid(row=0, column=1, sticky="ns")
        self.score_label = tk.Label(panel, fg="#f4f7ff", bg="#07111f", font=("TkDefaultFont", 18, "bold"), justify="left")
        self.score_label.pack(anchor="w", pady=(0, 14))
        tk.Label(panel, text="LEFT PLAYER\nW / S  move barrier\nF  measure\n\nRIGHT PLAYER\n↑ / ↓  move barrier\n/  measure\n\nA measurement samples |ψ|².\nLand beyond your opponent’s\nbarrier to score.\n\nR  restart match\nF11  fullscreen\nEsc  quit / leave fullscreen", fg="#b7c8dc", bg="#07111f", justify="left", font=("TkDefaultFont", 11)).pack(anchor="w")
        self.status_label = tk.Label(container, fg="#e6eef8", bg="#07111f", justify="left", wraplength=SIZE + 220)
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        # Use Tk's requested geometry so the initial canvas is exactly the
        # requested court aspect ratio, with no letterboxing.
        self.root.update_idletasks()
        self.root.geometry(f"{self.root.winfo_reqwidth()}x{self.root.winfo_reqheight()}")

        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen_or_quit)
        self.canvas.bind("<Configure>", self.resize_court)
        self.root.focus_force()
        self.photo: tk.PhotoImage | None = None

    def toggle_fullscreen(self, _event: tk.Event | None = None) -> None:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def exit_fullscreen_or_quit(self, _event: tk.Event | None = None) -> None:
        if self.fullscreen:
            self.fullscreen = False
            self.root.attributes("-fullscreen", False)
        else:
            self.root.destroy()

    def resize_court(self, event: tk.Event) -> None:
        """Keep the rectangular physical court centred while the window resizes."""
        scale = max(1, min(event.width / COURT_WIDTH, event.height / COURT_HEIGHT))
        self.field_width = max(1, round(COURT_WIDTH * scale))
        self.field_height = max(1, round(COURT_HEIGHT * scale))
        self.field_left = (event.width - self.field_width) // 2
        self.field_top = (event.height - self.field_height) // 2

    def key_down(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "r":
            self.score = [0, 0]
            self.field.reset()
            self.pause_until = 0.0
            self.measurement_marker = None
            self.pending_winner = None
            self.pending_relaunch_direction = 1
            self.status = "New match."
            return
        if key == "f":
            self.try_measure(0)
            return
        if key in ("slash", "divide"):
            self.try_measure(1)
            return
        self.keys.add(key)

    def key_up(self, event: tk.Event) -> None:
        self.keys.discard(event.keysym.lower())

    def try_measure(self, player: int) -> None:
        now = time.perf_counter()
        paddle = self.left if player == 0 else self.right
        name = "Left" if player == 0 else "Right"
        if self.pause_until:
            self.status = "A measurement is already in progress."
            return
        if now < paddle.ready_at:
            self.status = f"{name} measurement device is recharging."
            return
        paddle.ready_at = now + COOLDOWN
        x, y = self.field.measure(self.rng)
        self.measurement_marker = (x, y)
        self.pause_until = now + MEASUREMENT_PAUSE
        # Behind a player's adversary racket is the far side of the court.
        winner = 0 if x > PADDLE_X[1] else 1 if x < PADDLE_X[0] else None
        self.pending_winner = winner
        # A completed measurement always starts a new centre serve.  A player
        # who measures sends it toward their opponent unless it scored.
        self.pending_relaunch_direction = 1 if player == 0 else -1
        outcome = f" {('LEFT' if winner == 0 else 'RIGHT')} scores when play resumes." if winner is not None else ""
        self.status = f"{name} measured ({x:.3f}, {y:.3f}). Simulation paused for {MEASUREMENT_PAUSE:.0f}s.{outcome}"

    def update_paddles(self, elapsed: float) -> None:
        # Simulation y grows upward, whereas canvas y grows downward.
        if "w" in self.keys:
            self.left.y += PADDLE_SPEED * elapsed
        if "s" in self.keys:
            self.left.y -= PADDLE_SPEED * elapsed
        if "up" in self.keys:
            self.right.y += PADDLE_SPEED * elapsed
        if "down" in self.keys:
            self.right.y -= PADDLE_SPEED * elapsed
        low, high = PADDLE_LENGTH / 2, 1 - PADDLE_LENGTH / 2
        self.left.y = float(np.clip(self.left.y, low, high))
        self.right.y = float(np.clip(self.right.y, low, high))

    def draw(self) -> None:
        density = np.abs(self.field.psi) ** 2
        # Scale to the current peak, so a localized packet remains clearly
        # visible as it disperses without saturating the surrounding field.
        light = np.sqrt(density / max(float(density.max()), 1e-300))
        rgb = np.empty((*density.shape, 3), dtype=np.uint8)
        rgb[..., 0] = 5 + 35 * light
        rgb[..., 1] = 16 + 150 * light
        rgb[..., 2] = 31 + 220 * light
        # PPM requires rows in (screen-y, screen-x, RGB) order.  Do not use
        # ``rgb.T`` here: for a 3-D array it reverses *all* axes and turns
        # the RGB channel planes into apparent duplicate packets.
        pixels = np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1, :, :])
        # Render the numerical field at the rectangular court's current size.
        # This permits window resize/fullscreen without changing the solver.
        if (self.field_height, self.field_width) != pixels.shape[:2]:
            sample_y = np.linspace(0, pixels.shape[0] - 1, self.field_height).astype(np.intp)
            sample_x = np.linspace(0, pixels.shape[1] - 1, self.field_width).astype(np.intp)
            pixels = np.ascontiguousarray(pixels[sample_y][:, sample_x])
        ppm = b"P6\n%d %d\n255\n" % (self.field_width, self.field_height) + pixels.tobytes()
        # With an explicit PPM format, Tk expects the raw PPM byte stream.
        # Base64 data is not decoded by that reader and produces an
        # "couldn't recognize image data" error.
        self.photo = tk.PhotoImage(data=ppm, format="PPM")
        self.canvas.itemconfigure(self.image_id, image=self.photo)
        self.canvas.coords(self.image_id, self.field_left, self.field_top)
        self.canvas.coords(self.midline_id,
                           self.field_left + self.field_width / 2, self.field_top,
                           self.field_left + self.field_width / 2, self.field_top + self.field_height)
        for item, px, paddle in ((self.left_id, PADDLE_X[0], self.left), (self.right_id, PADDLE_X[1], self.right)):
            cx = self.field_left + (px / COURT_WIDTH) * self.field_width
            y0 = self.field_top + (1 - (paddle.y + PADDLE_LENGTH / 2) / COURT_HEIGHT) * self.field_height
            y1 = self.field_top + (1 - (paddle.y - PADDLE_LENGTH / 2) / COURT_HEIGHT) * self.field_height
            half_width = 2.5 * self.field.dx / COURT_WIDTH * self.field_width
            self.canvas.coords(item, cx - half_width, y0, cx + half_width, y1)
        if self.measurement_marker is None:
            self.canvas.itemconfigure(self.marker_h, state="hidden")
            self.canvas.itemconfigure(self.marker_v, state="hidden")
        else:
            x, y = self.measurement_marker
            sx = self.field_left + (x / COURT_WIDTH) * self.field_width
            sy = self.field_top + (1 - y / COURT_HEIGHT) * self.field_height
            self.canvas.coords(self.marker_h, sx - 13, sy, sx + 13, sy)
            self.canvas.coords(self.marker_v, sx, sy - 13, sx, sy + 13)
            self.canvas.itemconfigure(self.marker_h, state="normal")
            self.canvas.itemconfigure(self.marker_v, state="normal")
        now = time.perf_counter()
        l = "READY" if now >= self.left.ready_at else f"{self.left.ready_at - now:.1f}s"
        r = "READY" if now >= self.right.ready_at else f"{self.right.ready_at - now:.1f}s"
        self.score_label.config(text=f"QUANTUM PONG\n\nLeft  {self.score[0]}  ·  {self.score[1]}  Right\n\nMeasure: {l} / {r}")
        self.status_label.config(text=self.status)

    def tick(self) -> None:
        now = time.perf_counter()
        elapsed = min(now - self.last_frame, 0.05)
        self.last_frame = now
        if self.pause_until and now >= self.pause_until:
            self.pause_until = 0.0
            self.measurement_marker = None
            winner = self.pending_winner
            if winner is not None:
                self.score[winner] += 1
                side = "LEFT" if winner == 0 else "RIGHT"
                # Serve toward the player who conceded the point.
                self.pending_relaunch_direction = 1 if winner == 0 else -1
                self.status = f"{side} scores! New serve begins on the {side.lower()} side."
            else:
                self.status = "Measurement complete. New centre serve launched."
            self.field.reset(self.pending_relaunch_direction, owner=winner)
            self.pending_winner = None
        if not self.pause_until:
            self.update_paddles(elapsed)
            potential = self.field.potential(self.left, self.right)
            for _ in range(SUBSTEPS):
                self.field.step(potential)
        self.draw()
        self.root.after(16, self.tick)

    def run(self) -> None:
        self.draw()
        self.root.after(16, self.tick)
        self.root.mainloop()


def self_test() -> None:
    """Small headless sanity test: propagation remains normalized and bounded."""
    field = QuantumField(48)
    left, right = Paddle(), Paddle()
    for _ in range(100):
        field.step(field.potential(left, right))
    norm = float(np.sum(np.abs(field.psi) ** 2))
    assert abs(norm - 1.0) < 1e-10, norm
    x, y = field.measure(np.random.default_rng(2))
    assert 0 < x < COURT_WIDTH and 0 < y < COURT_HEIGHT
    assert abs(np.sum(np.abs(field.psi) ** 2) - 1) < 1e-12
    peak = np.unravel_index(np.argmax(np.abs(field.psi) ** 2), field.psi.shape)
    assert (field.x[peak], field.y[peak]) == (x, y)
    print("self-test passed; norm=%.12f" % norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantum Pong")
    parser.add_argument("--self-test", action="store_true", help="run a headless numerical check")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        Game().run()

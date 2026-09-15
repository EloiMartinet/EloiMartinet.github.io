import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ============================================================
# 1. PARAMETERS
# ============================================================

DX = 2e-2
N_K = 2 * int(1 / DX) + 1
DX = 1.0 / N_K
L = 6.0
N = int(L / DX)

DT = 2e-2
N_STEPS = 1000

# Movie
OUTPUT_FILE = "asymptotic_lenia.mp4"
SAVE_EVERY = 10
FPS = 30

# Lenia parameters
BETA = torch.tensor([1.0, 0.5])
MU = 0.3
SIGMA = 0.028

# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if INITIAL_FILE is not None:
    print(f"Loading initial condition: {INITIAL_FILE}")


# ============================================================
# 3. SPATIAL GRID
# ============================================================

x = torch.arange(N, device=device) * DX - L / 2
y = torch.arange(N, device=device) * DX - L / 2
X, Y = torch.meshgrid(x, y, indexing="ij")


# ============================================================
# 4. INITIAL CONDITION
# ============================================================

M = 10

MIN_RADIUS = 0.6
MAX_RADIUS = 0.9

u = torch.zeros((N, N), device=device)

centers = []
radii = []

for _ in range(M):

    radius = (
        MIN_RADIUS
        + (MAX_RADIUS - MIN_RADIUS)
        * torch.rand((), device=device)
    )

    cx = (
        torch.rand((), device=device)
        * (L - 2 * radius)
        - (L - 2 * radius) / 2
    )

    cy = (
        torch.rand((), device=device)
        * (L - 2 * radius)
        - (L - 2 * radius) / 2
    )

    centers.append(cx)
    centers.append(cy)
    radii.append(radius)

    disk = torch.zeros_like(u)

    for _ in range(20):

        kx = torch.randint(-20, 21, (), device=device)
        ky = torch.randint(-20, 21, (), device=device)

        k = torch.sqrt(kx**2 + ky**2)

        a = torch.randn((), device=device) / (1 + k)

        phi = 2 * torch.pi * torch.rand((), device=device)

        disk += a * torch.cos(
            2 * torch.pi * (kx * X / L + ky * Y / L) + phi
        )

    disk -= disk.amin()
    disk /= disk.amax() + 1e-12

    r = torch.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = r < radius

    u += disk * mask

u -= u.amin()
u /= u.amax() + 1e-12


# Add batch and channel dimensions
u = u[None, None]

# ============================================================
# 5. LENIA KERNEL
# ============================================================

def kernel_core(r):
    return torch.exp(4.0 - 1.0 / (r * (1.0 - r) + 1e-12))


def kernel_shell(r, beta):
    B = len(beta)
    kernel = torch.zeros_like(r)

    mask = (r >= 0.0) & (r < 1.0)
    rm = r[mask]

    shell_position = B * rm
    shell_index = torch.floor(shell_position).long()
    shell_r = shell_position % 1.0

    kernel[mask] = beta[shell_index] * kernel_core(shell_r)

    return kernel


kernel_x = (torch.arange(N_K, device=device) - N_K // 2) * DX
kernel_y = (torch.arange(N_K, device=device) - N_K // 2) * DX
KX, KY = torch.meshgrid(kernel_x, kernel_y, indexing="ij")
R = torch.sqrt(KX**2 + KY**2)

weight = kernel_shell(R, BETA.to(device))
weight[R >= 1.0] = 0.0
weight /= weight.sum()

print(f"Kernel shape : {weight.shape}")
print(f"Kernel sum   : {weight.sum().item():.6f}")
print(f"Kernel max   : {weight.max().item():.6f}")


# ============================================================
# 6. FFT CONVOLUTION
# ============================================================

kernel = torch.zeros((N, N), device=device)
kernel[:N_K, :N_K] = weight

kernel = torch.roll(
    kernel,
    shifts=(-(N_K // 2), -(N_K // 2)),
    dims=(0, 1)
)

kernel_fft = torch.fft.rfft2(kernel)


def conv_fft(u):
    return torch.fft.irfft2(
        torch.fft.rfft2(u) * kernel_fft,
        s=(N, N)
    )


# ============================================================
# 7. GROWTH FUNCTION
# ============================================================

def growth(z):
    return torch.exp(-0.5 * ((z - MU) / SIGMA) ** 2)


# ============================================================
# 8. VISUALIZATION
# ============================================================

plt.ioff()

fig, ax = plt.subplots(figsize=(10, 10))

u_cpu = u[0, 0].detach().cpu()

im = ax.imshow(
    u_cpu,
    extent=[-L / 2, L / 2, -L / 2, L / 2],
    origin="lower",
    cmap="inferno",
    vmin=0.0,
    vmax=1.0,
    interpolation="bilinear"
)

ax.set_axis_off()
ax.set_position([0, 0, 1, 1])
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


# ============================================================
# 9. VIDEO WRITER
# ============================================================

writer = imageio.get_writer(
    OUTPUT_FILE,
    fps=FPS,
    codec="libx264",
    quality=7
)


# ============================================================
# 10. TIME INTEGRATION
# ============================================================

print("\nStarting simulation...")

with torch.no_grad():
    for step in range(N_STEPS + 1):

        if step % SAVE_EVERY == 0:

            u_cpu = u[0, 0].detach().cpu()
            im.set_data(u_cpu)

            time = step * DT

            fig.canvas.draw()

            frame = torch.frombuffer(
                fig.canvas.buffer_rgba(),
                dtype=torch.uint8
            )

            frame = frame.reshape(
                fig.canvas.get_width_height()[::-1] + (4,)
            )

            frame = frame[:, :, :3].numpy()
            writer.append_data(frame)

            print(
                f"step {step:6d}/{N_STEPS} | "
                f"t = {time:8.3f}"
            )

        perception = conv_fft(u)
        growth_field = growth(perception)
        u = u + DT * (growth_field - u)


# ============================================================
# 11. CLEANUP
# ============================================================

writer.close()
plt.close(fig)

print("\nSimulation finished.")
print(f"Movie saved to: {os.path.abspath(OUTPUT_FILE)}")
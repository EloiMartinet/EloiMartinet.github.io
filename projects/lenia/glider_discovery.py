import torch
import matplotlib.pyplot as plt
import matplotlib

# torch.set_default_dtype(torch.float64)

# ============================================================
# 1. PARAMETERS
# ============================================================

DX = 3e-2

N_K = 2 * int(1 / DX) + 1
DX = 1.0 / N_K

L = 2.0
N = int(L / DX)

LOC_RADIUS = 0.5
LOC_WIDTH = 0.01       
LOC_LAMBDA = 0.1

# ------------------------------------------------------------
# Lenia parameters
# ------------------------------------------------------------

BETA = torch.tensor([0.1, 0.5, 0.8, 1.0])

MU = 0.25
SIGMA = 0.05

# ============================================================
# OPTIMIZABLE LENIA PARAMETERS
# ============================================================

MU_MIN = 0.01
MU_MAX = 0.50

SIGMA_MIN = 0.02
SIGMA_MAX = 0.3

# ------------------------------------------------------------
# Glider velocity
# ------------------------------------------------------------

VX = 0.04
VY = 0.04

# ------------------------------------------------------------
# LBFGS parameters
# ------------------------------------------------------------

ADAM_STEPS = 300
ADAM_LR = 0.02

LBFGS_STEPS = 700
LBFGS_MAX_ITER = 20
LR_LBFGS = 20000000.0

# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ============================================================
# 18. SAVE OPTIMAL PROFILE
# ============================================================

import numpy as np


def save_optimal_profile(
    filename,
    u,
    mu,
    sigma,
    beta,
    dx,
):
    """
    Save the optimized Lenia profile and all relevant constants.

    The file is stored as a NumPy .npz archive and can be read
    easily from another Python script using:

        data = np.load(filename)

        u = data["u"]
        mu = float(data["mu"])
        sigma = float(data["sigma"])
        beta = data["beta"]

    Parameters
    ----------
    filename : str
        Output .npz filename.

    u : torch.Tensor
        Optimized profile with shape (1, 1, N, N).

    mu, sigma : float
        Lenia growth-function parameters.

    beta : torch.Tensor
        Lenia kernel shell weights.

    dx : float
        Spatial grid spacing.
    """

    # Remove batch/channel dimensions.
    u_numpy = (
        u.detach()
        .cpu()
        .numpy()
        .squeeze()
        .astype(np.float64)
    )

    beta_numpy = (
        beta.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    np.savez(
        filename,
        u=u_numpy,
        mu=np.float64(mu),
        sigma=np.float64(sigma),
        beta=beta_numpy,
        dx=np.float64(dx),
    )

    print()
    print("Saved optimal profile:")
    print("  file         =", filename)
    print("  u shape      =", u_numpy.shape)
    print("  mu           =", mu)
    print("  sigma        =", sigma)
    print("  beta         =", beta_numpy)
    print("  dx           =", dx)



# ============================================================
# 3. SPATIAL GRID
# ============================================================

x = torch.arange(N, device=device) * DX - L / 2
y = torch.arange(N, device=device) * DX - L / 2
X, Y = torch.meshgrid(x, y, indexing="ij")

# ============================================================
# LOCALIZATION PENALTY
# ============================================================

R2 = X**2 + Y**2
R = torch.sqrt(R2)

# Smooth approximation of:
#
#       0                  r < LOC_RADIUS
#       1                  r > LOC_RADIUS
#
localization_weight = torch.sigmoid((R - LOC_RADIUS) / LOC_WIDTH)

# Add batch/channel dimensions
localization_weight = localization_weight[None, None]

# ============================================================
# 4. STATIC INITIAL GUESS
# ============================================================

u = torch.ones_like(X)

for _ in range(20):

    kx0 = torch.randint(-8, 9, (), device=device)
    ky0 = torch.randint(-8, 9, (), device=device)
    k = torch.sqrt(kx0**2 + ky0**2)

    amplitude = torch.randn((), device=device) / (1.0 + k)

    phase = 2.0 * torch.pi * torch.rand((), device=device)
    arg = 2.0 * torch.pi * ( kx0 * X / L + ky0 * Y / L )  + phase
    u += amplitude * torch.cos(arg)


u = u[None, None] 
u -= 5*localization_weight

# Normalize texture
u -= u.amin()
u /= u.amax()

# Add batch/channel dimensions
u = u * (1 - localization_weight)


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


# ------------------------------------------------------------
# Kernel coordinates
# ------------------------------------------------------------

kernel_x = ( torch.arange(N_K, device=device) - N_K // 2 )* DX
kernel_y = ( torch.arange(N_K, device=device) - N_K // 2 )* DX
KX, KY = torch.meshgrid(kernel_x, kernel_y, indexing="ij")
R = torch.sqrt(KX**2 + KY**2)


# ------------------------------------------------------------
# Construct kernel
# ------------------------------------------------------------

weight = kernel_shell(R, BETA.to(device))
weight[R >= 1.0] = 0.0
weight /= weight.sum()

print()
print("Kernel:")
print("  shape =", weight.shape)
print("  sum   =", weight.sum().item())
print("  max   =", weight.max().item())


# ============================================================
# 6. FFT KERNEL
# ============================================================

kernel = torch.zeros((N, N), device=device)

kernel[:N_K, :N_K] = weight

kernel = torch.roll(
    kernel,
    shifts=(-(N_K // 2), -(N_K // 2) ),
    dims=(0, 1)
)

kernel_fft = torch.fft.rfft2(kernel)


# ============================================================
# 7. FOURIER WAVE NUMBERS
# ============================================================

kx = 2.0 * torch.pi * torch.fft.fftfreq(N, d=DX, device=device)
ky = 2.0 * torch.pi * torch.fft.rfftfreq(N, d=DX, device=device)
KX_F, KY_F = torch.meshgrid(kx, ky, indexing="ij")


# ============================================================
# 8. FFT OPERATORS
# ============================================================

def conv_fft(u):
    u_hat = torch.fft.rfft2(u)

    return torch.fft.irfft2(u_hat * kernel_fft, s=(N, N))


def gradient_fft(u):
    u_hat = torch.fft.rfft2(u)
    ux = torch.fft.irfft2(1j * KX_F * u_hat, s=(N, N))
    uy = torch.fft.irfft2(1j * KY_F * u_hat, s=(N, N))

    return ux, uy

# ============================================================
# FINITE-DIFFERENCE GRADIENT OPERATOR
# ============================================================

def gradient_fd(u):
    """
    Centered finite differences with periodic boundary conditions.

    du/dx ≈ [u(x+dx) - u(x-dx)] / (2 dx)
    du/dy ≈ [u(y+dy) - u(y-dy)] / (2 dx)
    """

    ux = (
        torch.roll(u, shifts=-1, dims=-2)
        - torch.roll(u, shifts=+1, dims=-2)
    ) / (2.0 * DX)

    uy = (
        torch.roll(u, shifts=-1, dims=-1)
        - torch.roll(u, shifts=+1, dims=-1)
    ) / (2.0 * DX)

    return ux, uy


# ============================================================
# 9. GROWTH FUNCTION
# ============================================================

def growth(z):
    mu = get_mu()
    sigma = get_sigma()

    return torch.exp(-0.5 * ((z - mu) / sigma) ** 2)

# ============================================================
# DISCRETE LENIA DYNAMICS
# ============================================================

def lenia_step(u):
    perception = conv_fft(u)
    growth_field = growth(perception)
    return u + DT * (growth_field - u)


# ============================================================
# FOURIER TRANSLATION
# ============================================================

def translate_fft(u, dx_shift, dy_shift):
    u_hat = torch.fft.rfft2(u)

    phase = torch.exp(
        -1j * (
            KX_F * dx_shift +
            KY_F * dy_shift
        )
    )

    return torch.fft.irfft2(
        u_hat * phase,
        s=(N, N)
    )


# ============================================================
# DISCRETE GLIDER RESIDUAL
# ============================================================

def glider_residual(u):
    # T(G*u)
    growth_field = growth(conv_fft(u))

    # v . grad U
    ux, uy = gradient_fd(u)
    # ux, uy = gradient_fft(u)
    transport = VX * ux + VY * uy

    return u - transport - growth_field


# ============================================================
# MASS
# ============================================================

def total_mass(u):
    return torch.sum(u, dim=(-2, -1)) * DX**2


# ============================================================
# LOSS
# ============================================================


def loss_function():
    u = get_u()

    residual = glider_residual(u)
    loss_residual = 0.5 * torch.mean(residual**2)

    total_mass = torch.mean(u)
    outside_mass = torch.mean(localization_weight * u)
    loss_localization = outside_mass / (total_mass + 1e-12)

    loss =  loss_residual + LOC_LAMBDA * loss_localization

    return loss

# ============================================================
# 12. SIGMOID PARAMETRIZATION
# ============================================================

eps = 1e-3

u_initial = u.detach().clone()
u_initial = u_initial.clamp(eps, 1.0 - eps)

q_opt = torch.log(
    u_initial / (1.0 - u_initial)
)
q_opt.requires_grad_(True)


# ============================================================
# OPTIMIZABLE MU AND SIGMA
# ============================================================

def inverse_sigmoid(x):
    return torch.log(x / (1.0 - x))


# ------------------------------------------------------------
# MU
#
# mu = MU_MIN + sigmoid(q_mu) * (MU_MAX - MU_MIN)
# ------------------------------------------------------------

mu_normalized = (
    (MU - MU_MIN) /
    (MU_MAX - MU_MIN)
)

q_mu = inverse_sigmoid(
    torch.tensor(mu_normalized, device=device)
)

q_mu.requires_grad_(True)


# ------------------------------------------------------------
# SIGMA
#
# sigma = SIGMA_MIN + sigmoid(q_sigma)
#                       * (SIGMA_MAX - SIGMA_MIN)
# ------------------------------------------------------------

sigma_normalized = (
    (SIGMA - SIGMA_MIN) /
    (SIGMA_MAX - SIGMA_MIN)
)

q_sigma = inverse_sigmoid(
    torch.tensor(sigma_normalized, device=device)
)

q_sigma.requires_grad_(True)


# ============================================================
# OPTIMIZED PARAMETERS
# ============================================================

def get_u():
    return torch.sigmoid(q_opt)


def get_mu():
    return (
        MU_MIN
        + (MU_MAX - MU_MIN) * torch.sigmoid(q_mu)
    )


def get_sigma():
    return (
        SIGMA_MIN
        + (SIGMA_MAX - SIGMA_MIN) * torch.sigmoid(q_sigma)
    )

def closure():
    optimizer.zero_grad()
    loss = loss_function()
    loss.backward()

    return loss

def step(iteration, print_every=20):
    if iteration % print_every == 0:

        with torch.no_grad():
            loss = loss_function()
            u_current = get_u()
            residual = glider_residual(u_current)
            rms = torch.sqrt(torch.mean(residual**2))

            u_min = u_current.min()
            u_max = u_current.max()

            mu_current = get_mu()
            sigma_current = get_sigma()

        print(
            f"iteration {iteration:5d} | "
            f"loss = {loss.item():.8e} | "
            f"RMS = {rms.item():.8e} | "
            f"mu = {mu_current.item():.8f} | "
            f"sigma = {sigma_current.item():.8f} | "
            f"U range = "
            f"[{u_min.item():.5f}, "
            f"{u_max.item():.5f}]"
        )

                
        plt.figure(figsize=(8, 8))

        u = u_current[0, 0].detach().cpu().numpy()

        # Get colormap using the modern Matplotlib API
        cmap = matplotlib.colormaps["jet"]

        # Normalize u from [0, 1]
        rgba = cmap(u)

        # Set alpha equal to u:
        # u=0 -> transparent, u=1 -> opaque
        rgba[..., 3] = u**0.5

        plt.imshow(
            rgba,
            extent=[-L / 2, L / 2, -L / 2, L / 2],
            origin="lower",
            interpolation="bilinear"
        )

        plt.axis("off")
        plt.tight_layout()

        plt.savefig(
            f"res/glider_{iteration}.png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

    optimizer.step(closure)

# ============================================================
# 15. ADAM OPTIM
# ============================================================

optimizer = torch.optim.Adam([q_opt, q_mu, q_sigma], lr=ADAM_LR)


for iteration in range(ADAM_STEPS):
    step(iteration, print_every=10)

# ============================================================
# 15. LBFGS OPTIM
# ============================================================

# Deactivate the localization penalty
def loss_function():
    u = get_u()

    residual = glider_residual(u)
    loss = 0.5 * torch.mean(residual**2)

    return loss

optimizer = torch.optim.LBFGS(
    [q_opt, q_mu, q_sigma],
    lr=LR_LBFGS,                  # Standard default learning rate for LBFGS with line search
    max_iter=20,
    history_size=100,
    line_search_fn="strong_wolfe",  # <--- UNCOMMENT THIS
    tolerance_grad=1e-32,     # <--- Relaxed from 1e-15
    tolerance_change=1e-32,   # <--- Relaxed from 1e-15
)


for iteration in range(LBFGS_STEPS):
    step(iteration, print_every=10)



# ============================================================
# 17. FINAL PROFILE
# ============================================================

with torch.no_grad():
    u_final = get_u()
    mu_final = get_mu()
    sigma_final = get_sigma()

    residual = glider_residual(u_final)

    loss = 0.5 * torch.mean(residual**2)

    rms = torch.sqrt(torch.mean(residual**2))

    max_residual = residual.abs().max()

    u_min = u_final.min()
    u_max = u_final.max()


print()
print("Final optimized parameters:")
print("  mu    =", mu_final.item())
print("  sigma =", sigma_final.item())
print("  loss  =", loss.item())
print("  RMS   =", rms.item())
print("  max residual =", max_residual.item())


# ============================================================
# SAVE FINAL RESULT
# ============================================================

save_optimal_profile(
    filename="res/optimal_glider.npz",
    u=u_final,
    mu=mu_final.item(),
    sigma=sigma_final.item(),
    beta=BETA,
    dx=DX,
)



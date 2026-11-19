"""
velocity_field.py
=================
Purpose : Compute the 2-D velocity field around the airfoil using thin airfoil
          theory (vortex-sheet model) and evaluate total circulation via a
          closed-path velocity line integral.

Inputs  : Camber line function, Fourier coefficients (A0, An),
          freestream velocity, angle of attack, domain parameters.

Outputs : Velocity component arrays on a grid; total circulation Γ.

Theory
------
Each infinitesimal vortex element at position (ξ, ζ) on the camber line with
circulation strength dΓ = γ(ξ) dξ induces a velocity at field point (x, z):

    du =  dΓ/(2π) · (z − ζ) / r²
    dw = −dΓ/(2π) · (x − ξ) / r²

where  r² = (x−ξ)² + (z−ζ)².  (2-D Biot–Savart law for a line vortex)

Total velocity = freestream + Σ induced velocities from all vortex elements.

Circulation via line integral (Stokes' theorem):
    Γ = ∮_C V · dl   (counterclockwise path enclosing the airfoil)

Assumptions
-----------
* Vortex elements lie along the camber line (thin airfoil approximation).
* Incompressible, inviscid, 2-D potential flow.
* Grid memory is managed by using float32 arrays and batched computation.
"""

import numpy as np
from aerodynamics import compute_fourier_coefficients, compute_gamma_distribution


# ---------------------------------------------------------------------------
# Discrete vortex elements
# ---------------------------------------------------------------------------

def get_vortex_elements(camber_func, A0: float, An: np.ndarray,
                         V_inf: float, chord: float = 1.0,
                         n_vortex: int = 300):
    """
    Discretise the continuous vortex sheet into N point vortices.

    Uses cosine spacing in θ-space (denser near LE and TE).

    Parameters
    ----------
    camber_func : callable    z = f(x) camber line
    A0          : float       Zeroth Fourier coefficient
    An          : ndarray     Higher Fourier coefficients (A1 … AN)
    V_inf       : float       Freestream velocity [m/s]
    chord       : float       Chord length [m]
    n_vortex    : int         Number of discrete vortex elements

    Returns
    -------
    xi        : ndarray (N,)   x-positions of vortex elements
    zi        : ndarray (N,)   z-positions (on camber line)
    gamma_ds  : ndarray (N,)   Circulation of each element  γ(θ)·(c/2)sinθ·dθ  [m²/s]
    """
    dtheta = np.pi / n_vortex
    # Midpoints of θ intervals — avoids the leading-edge singularity at θ=0
    theta = np.linspace(dtheta / 2.0, np.pi - dtheta / 2.0, n_vortex)

    # Chord-position of each vortex element
    xi = (chord / 2.0) * (1.0 - np.cos(theta))

    # Height of each element (on camber line)
    try:
        zi = np.asarray(camber_func(xi), dtype=float)
    except Exception:
        zi = np.array([float(camber_func(x)) for x in xi])

    # Vortex-sheet strength  γ(θ)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    gamma_A0 = A0 * (1.0 + cos_t) / sin_t
    gamma_An = sum(a_n * np.sin((n + 1) * theta) for n, a_n in enumerate(An))
    gamma_val = 2.0 * V_inf * (gamma_A0 + gamma_An)

    # Arc-length element  ds = (c/2) sinθ dθ
    ds = (chord / 2.0) * sin_t * dtheta

    gamma_ds = gamma_val * ds   # circulation per element [m²/s]

    return xi, zi, gamma_ds


# ---------------------------------------------------------------------------
# Induced velocity (vectorised Biot-Savart)
# ---------------------------------------------------------------------------

def _induced_velocity_batch(eval_x: np.ndarray, eval_z: np.ndarray,
                             xi: np.ndarray, zi: np.ndarray,
                             gamma_ds: np.ndarray):
    """
    Compute induced velocity at multiple field points simultaneously.

    Parameters
    ----------
    eval_x, eval_z : ndarray (M,)   Field point coordinates
    xi, zi         : ndarray (N,)   Vortex element positions
    gamma_ds       : ndarray (N,)   Vortex element circulations

    Returns
    -------
    u, w : ndarray (M,)   Induced velocity components (x, z)
    """
    # Difference arrays: shape (M, N)
    dx = eval_x[:, np.newaxis] - xi[np.newaxis, :]  # (M, N)
    dz = eval_z[:, np.newaxis] - zi[np.newaxis, :]

    r2 = dx**2 + dz**2
    r2 = np.maximum(r2, 1e-9)   # prevent division by zero

    # Biot-Savart:  du = dΓ/(2π) · dz/r²
    u = np.sum(gamma_ds[np.newaxis, :] * dz / (2.0 * np.pi * r2), axis=1)
    w = np.sum(-gamma_ds[np.newaxis, :] * dx / (2.0 * np.pi * r2), axis=1)

    return u, w


# ---------------------------------------------------------------------------
# Full velocity field on a grid
# ---------------------------------------------------------------------------

def compute_velocity_field(camber_func, A0: float, An: np.ndarray,
                            V_inf: float, alpha_rad: float,
                            chord: float = 1.0,
                            n_vortex: int = 250,
                            nx: int = 55, nz: int = 42):
    """
    Compute the 2-D velocity vector field on a grid.

    Domain : 4c × 3c   (x ∈ [−0.5c, 3.5c],  z ∈ [−1.5c, 1.5c])

    Parameters
    ----------
    camber_func : callable    Camber line z = f(x)
    A0          : float       Zeroth Fourier coefficient
    An          : ndarray     Higher Fourier coefficients
    V_inf       : float       Freestream velocity [m/s]
    alpha_rad   : float       Angle of attack [rad]
    chord       : float       Chord length [m]
    n_vortex    : int         Discrete vortex resolution
    nx, nz      : int         Grid resolution (x and z directions)

    Returns
    -------
    X  : ndarray (nz, nx)   Grid x-coordinates
    Z  : ndarray (nz, nx)   Grid z-coordinates
    Vx : ndarray (nz, nx)   x-velocity component [m/s]  (NaN on camber line)
    Vz : ndarray (nz, nx)   z-velocity component [m/s]
    Vmag : ndarray (nz, nx) Velocity magnitude [m/s]
    """
    xi, zi, gamma_ds = get_vortex_elements(camber_func, A0, An, V_inf, chord, n_vortex)

    # Grid spanning 4c × 3c
    x_lin = np.linspace(-0.5 * chord, 3.5 * chord, nx)
    z_lin = np.linspace(-1.5 * chord, 1.5 * chord, nz)
    X, Z = np.meshgrid(x_lin, z_lin)

    # Flatten grid to 1-D for vectorised computation
    eval_x = X.ravel().astype(np.float64)
    eval_z = Z.ravel().astype(np.float64)

    u_ind, w_ind = _induced_velocity_batch(eval_x, eval_z, xi, zi, gamma_ds)

    # Total velocity = freestream + induced
    Vx_flat = V_inf * np.cos(alpha_rad) + u_ind
    Vz_flat = V_inf * np.sin(alpha_rad) + w_ind

    Vx = Vx_flat.reshape(nz, nx)
    Vz = Vz_flat.reshape(nz, nx)

    # Mask grid points very close to the camber line (avoid artefacts)
    try:
        Z_camber = np.where((X >= 0) & (X <= chord),
                             camber_func(np.clip(X, 0, chord)), np.nan)
    except Exception:
        Z_camber_vals = np.array([float(camber_func(np.clip(xv, 0, chord)))
                                   if 0 <= xv <= chord else np.nan
                                   for xv in X.ravel()])
        Z_camber = Z_camber_vals.reshape(nz, nx)

    mask = (X >= 0) & (X <= chord) & (np.abs(Z - Z_camber) < 0.025 * chord)
    Vx = np.where(mask, np.nan, Vx)
    Vz = np.where(mask, np.nan, Vz)

    Vmag = np.sqrt(Vx**2 + Vz**2)
    return X, Z, Vx, Vz, Vmag


# ---------------------------------------------------------------------------
# Circulation via velocity line integral (Γ = ∮ V · dl)
# ---------------------------------------------------------------------------

def compute_circulation_line_integral(camber_func, A0: float, An: np.ndarray,
                                       V_inf: float, alpha_rad: float,
                                       chord: float = 1.0,
                                       n_vortex: int = 250,
                                       n_pts_per_side: int = 300):
    """
    Compute bound circulation using a rectangular velocity line integral.

    Path : counterclockwise rectangle surrounding the airfoil, with margins
           of 0.4c on each side.  Γ = ∮_C V · dl

    Parameters
    ----------
    camber_func     : callable    Camber line z = f(x)
    A0, An          : Fourier coefficients
    V_inf           : float       Freestream velocity [m/s]
    alpha_rad       : float       Angle of attack [rad]
    chord           : float       Chord length [m]
    n_vortex        : int         Discrete vortex elements
    n_pts_per_side  : int         Sample points per side of the rectangle

    Returns
    -------
    float   Total circulation Γ [m²/s]  (positive = counterclockwise = lift-producing)
    """
    xi, zi, gamma_ds = get_vortex_elements(camber_func, A0, An, V_inf, chord, n_vortex)

    margin = 0.4 * chord
    x_L = -margin                       # left boundary
    x_R = chord + margin                # right boundary
    z_B = -margin - 0.3 * chord         # bottom boundary
    z_T =  margin + 0.3 * chord         # top boundary

    freestream_u = V_inf * np.cos(alpha_rad)
    freestream_w = V_inf * np.sin(alpha_rad)

    def velocity_on_segment(pts_x, pts_z):
        """Total velocity (freestream + induced) at an array of points."""
        u_ind, w_ind = _induced_velocity_batch(pts_x, pts_z, xi, zi, gamma_ds)
        return freestream_u + u_ind, freestream_w + w_ind

    n = n_pts_per_side
    gamma_total = 0.0

    # --- Bottom: left → right,  z = z_B,  dl = (dx, 0) ---
    pts_x = np.linspace(x_L, x_R, n)
    pts_z = np.full(n, z_B)
    u, _ = velocity_on_segment(pts_x, pts_z)
    # Trapezoidal rule along x
    gamma_total += np.trapezoid(u, pts_x)

    # --- Right: bottom → top,  x = x_R,  dl = (0, dz) ---
    pts_x = np.full(n, x_R)
    pts_z = np.linspace(z_B, z_T, n)
    _, w = velocity_on_segment(pts_x, pts_z)
    gamma_total += np.trapezoid(w, pts_z)

    # --- Top: right → left,  z = z_T,  dl = (−dx, 0) ---
    pts_x = np.linspace(x_R, x_L, n)
    pts_z = np.full(n, z_T)
    u, _ = velocity_on_segment(pts_x, pts_z)
    gamma_total += np.trapezoid(u, pts_x)   # trapz handles sign automatically

    # --- Left: top → bottom,  x = x_L,  dl = (0, −dz) ---
    pts_x = np.full(n, x_L)
    pts_z = np.linspace(z_T, z_B, n)
    _, w = velocity_on_segment(pts_x, pts_z)
    gamma_total += np.trapezoid(w, pts_z)

    # Sign convention: the code uses the engineering (CW-positive) convention for
    # gamma_ds, consistent with Kutta-Joukowski where positive Gamma = upward lift.
    # The contour above is traversed CCW; for a CW vortex sheet the CCW line
    # integral returns -Gamma.  Negate to return the positive bound circulation.
    return -gamma_total

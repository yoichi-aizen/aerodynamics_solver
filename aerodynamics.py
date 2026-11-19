"""
aerodynamics.py
===============
Purpose : Implement Thin Airfoil Theory (TAT) to compute:
            - Camber line slope  dz/dx
            - Fourier coefficients  A0, A1, A2, … AN
            - Section lift coefficient  Cl
            - Section moment coefficient  Cm  (about quarter-chord)
            - Vortex / circulation distribution  γ(θ)
            - Total bound circulation  Γ  (by integrating γ)
            - Alpha-sweep results for plotting curves

Inputs  : Camber line function  z = f(x), angle of attack α [rad], flow speed.

Outputs : Scalars and arrays as documented per function.

Theory (Anderson "Fundamentals of Aerodynamics", Ch.4)
-------------------------------------------------------
Variable substitution:  x = (c/2)(1 − cosθ),   θ ∈ [0, π]

Fourier representation of the vortex-sheet strength:
   γ(θ) = 2 V∞ [ A0 (1+cosθ)/sinθ  +  Σ_{n=1}^{N} An sin(nθ) ]

Boundary condition (flow tangency on camber line):
   A0 = α − (1/π) ∫₀^π (dz/dx) dθ
   An = (2/π) ∫₀^π (dz/dx) cos(nθ) dθ

Aerodynamic coefficients:
   Cl      = π (2 A0 + A1)
   Cm_c/4  = (π/4)(A2 − A1)          [moment about quarter-chord]
   Cm_LE   = −(π/2)(A0 + A1 − A2/2)  [moment about leading edge]

Bound circulation:
   Γ = ∫₀^c γ(x) dx = π c V∞ (A0 + A1/2) = c V∞ Cl / 2

Assumptions
-----------
* Incompressible, inviscid, irrotational, 2-D flow.
* Small angles of attack (linear theory – no stall).
* Airfoil is thin (camber << chord).
* Trailing-edge Kutta condition: γ(c) = 0  ↔  θ = π.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Slope of the camber line (central finite difference)
# ---------------------------------------------------------------------------

def compute_slope(camber_func, x, dx: float = 1e-5):
    """
    Compute dz/dx at position(s) x using the central-difference formula.

    Parameters
    ----------
    camber_func : callable   z = f(x), camber line
    x           : float or ndarray   Chord position(s) in [0, 1]
    dx          : float   Step size for finite difference

    Returns
    -------
    float or ndarray   dz/dx at the given position(s)
    """
    x = np.asarray(x, dtype=float)
    # Clamp forward / backward points to (0, 1) to stay on the airfoil
    xp = np.clip(x + dx, 1e-7, 1.0 - 1e-7)
    xm = np.clip(x - dx, 1e-7, 1.0 - 1e-7)
    step = xp - xm  # actual step (may differ from 2*dx near boundaries)

    try:
        return (camber_func(xp) - camber_func(xm)) / step
    except Exception:
        # Fallback: element-wise if camber_func doesn't vectorise
        zp = np.array([float(camber_func(xi)) for xi in np.atleast_1d(xp)])
        zm = np.array([float(camber_func(xi)) for xi in np.atleast_1d(xm)])
        result = (zp - zm) / np.atleast_1d(step)
        return float(result[0]) if x.ndim == 0 else result


# ---------------------------------------------------------------------------
# Fourier coefficients A0, An
# ---------------------------------------------------------------------------

def compute_fourier_coefficients(camber_func, alpha_rad: float,
                                  chord: float = 1.0,
                                  n_terms: int = 20,
                                  n_quad: int = 1000):
    """
    Compute Fourier coefficients for the thin-airfoil vortex distribution.

    Uses midpoint quadrature in θ-space to avoid endpoint singularities.

    Parameters
    ----------
    camber_func : callable   z = f(x), normalised camber line
    alpha_rad   : float      Angle of attack [radians]
    chord       : float      Chord length (default 1.0)
    n_terms     : int        Number of An terms to compute (A1 … AN)
    n_quad      : int        Quadrature resolution

    Returns
    -------
    dict with keys:
        'A0' : float          Zeroth Fourier coefficient
        'An' : ndarray (N,)   A1 … AN  (index 0 → A1, index 1 → A2, …)
    """
    dtheta = np.pi / n_quad
    # Midpoint rule: avoids θ=0 and θ=π exactly
    theta = np.linspace(dtheta / 2.0, np.pi - dtheta / 2.0, n_quad)

    # Map θ → x  (cosine spacing gives denser points near LE and TE)
    x_vals = (chord / 2.0) * (1.0 - np.cos(theta))

    # Slope at all quadrature points
    slope = compute_slope(camber_func, x_vals)

    # A0: determines lift at α = 0 offset (zero-lift angle contribution)
    A0 = alpha_rad - (1.0 / np.pi) * np.sum(slope) * dtheta

    # An (n = 1 … N): shape-dependent Fourier series coefficients
    An = np.zeros(n_terms)
    for n in range(1, n_terms + 1):
        An[n - 1] = (2.0 / np.pi) * np.sum(slope * np.cos(n * theta)) * dtheta

    return {'A0': A0, 'An': An}


# ---------------------------------------------------------------------------
# Lift and moment coefficients
# ---------------------------------------------------------------------------

def compute_Cl(A0: float, A1: float) -> float:
    """
    Section lift coefficient from thin airfoil theory.

    Cl = π (2 A0 + A1)

    Derivation: integrate γ(θ) over chord and apply Kutta-Joukowski theorem.
    The slope of Cl vs α is always 2π (rad⁻¹) regardless of camber shape.
    """
    return np.pi * (2.0 * A0 + A1)


def compute_Cm_quarter_chord(A1: float, A2: float) -> float:
    """
    Section moment coefficient about the quarter-chord point.

    Cm_c/4 = (π/4)(A2 − A1)

    Derivation:
        Cm_LE = −(π/2)(A0 + A1 − A2/2)    [about leading edge]
        Cm_c/4 = Cm_LE + Cl/4
               = −(π/2)(A0+A1−A2/2) + (π/4)(2A0+A1)
               = (π/4)(A2 − A1)

    The quarter-chord is the aerodynamic centre for symmetric airfoils
    (Cm_c/4 = 0). For cambered airfoils, Cm_c/4 depends only on shape
    (A1, A2) and is constant with angle of attack.
    """
    return (np.pi / 4.0) * (A2 - A1)


# ---------------------------------------------------------------------------
# Circulation distribution γ(θ)
# ---------------------------------------------------------------------------

def compute_gamma_distribution(A0: float, An: np.ndarray,
                                V_inf: float,
                                chord: float = 1.0,
                                n_points: int = 500):
    """
    Compute the vortex-sheet (circulation) distribution along the chord.

    γ(θ) = 2 V∞ [ A0 (1+cosθ)/sinθ  +  Σ_n An sin(nθ) ]

    The leading edge (θ=0) has an integrable singularity (γ → ∞);
    the trailing edge (θ=π) satisfies γ = 0 (Kutta condition).

    Parameters
    ----------
    A0      : float      Zeroth Fourier coefficient
    An      : ndarray    A1 … AN (index 0 → A1)
    V_inf   : float      Freestream velocity [m/s]
    chord   : float      Chord length [m]
    n_points: int        Evaluation points (endpoints excluded)

    Returns
    -------
    theta_arr : ndarray (n_points,)   θ values [rad]
    x_arr     : ndarray (n_points,)   Chord positions  x = (c/2)(1−cosθ)
    gamma_arr : ndarray (n_points,)   γ(θ) values [m/s]
    """
    # Avoid θ=0 (LE singularity) by starting slightly inside
    theta_arr = np.linspace(np.pi / n_points, np.pi * (n_points - 1) / n_points, n_points)

    sin_t = np.sin(theta_arr)
    cos_t = np.cos(theta_arr)

    # A0 term: (1 + cosθ)/sinθ — finite for all θ > 0
    gamma_A0 = A0 * (1.0 + cos_t) / sin_t

    # An terms: Σ An sin(nθ)
    gamma_An = np.zeros_like(theta_arr)
    for n, a_n in enumerate(An, start=1):
        gamma_An += a_n * np.sin(n * theta_arr)

    gamma_arr = 2.0 * V_inf * (gamma_A0 + gamma_An)
    x_arr = (chord / 2.0) * (1.0 - cos_t)

    return theta_arr, x_arr, gamma_arr


# ---------------------------------------------------------------------------
# Bound circulation (analytical integral)
# ---------------------------------------------------------------------------

def compute_bound_circulation(A0: float, A1: float,
                               V_inf: float,
                               chord: float = 1.0) -> float:
    """
    Total bound circulation by integrating the vortex distribution.

    Γ = ∫₀^c γ(x) dx
      = ∫₀^π γ(θ) · (c/2) sinθ dθ
      = π c V∞ (A0 + A1/2)

    Consistency check via Kutta-Joukowski:
      L = ρ V∞ Γ  →  Cl = 2Γ/(V∞ c)  →  Γ = c V∞ Cl / 2  ✓

    Parameters
    ----------
    A0, A1  : Fourier coefficients
    V_inf   : freestream velocity [m/s]
    chord   : chord length [m]

    Returns
    -------
    float   Total bound circulation Γ [m²/s]
    """
    return np.pi * chord * V_inf * (A0 + A1 / 2.0)


# ---------------------------------------------------------------------------
# Alpha sweep (vectorised over angles)
# ---------------------------------------------------------------------------

def sweep_alpha(camber_func, alpha_deg_min: float = -3.0,
                alpha_deg_max: float = 12.0,
                n_alpha: int = 50,
                V_inf: float = 1.0,
                chord: float = 1.0,
                n_quad: int = 1000):
    """
    Compute aerodynamic coefficients over a range of angles of attack.

    Parameters
    ----------
    camber_func   : callable   Camber line z = f(x)
    alpha_deg_min : float      Start angle [deg]
    alpha_deg_max : float      End angle [deg]
    n_alpha       : int        Number of alpha values
    V_inf         : float      Freestream velocity [m/s]
    chord         : float      Chord length [m]
    n_quad        : int        Quadrature points for Fourier integrals

    Returns
    -------
    dict with arrays:
        'alpha_deg' : angles in degrees
        'A0', 'A1', 'A2' : Fourier coefficients vs alpha
        'Cl', 'Cm' : aerodynamic coefficients vs alpha
    """
    alpha_deg = np.linspace(alpha_deg_min, alpha_deg_max, n_alpha)
    alpha_rad = np.deg2rad(alpha_deg)

    A0_arr = np.zeros(n_alpha)
    A1_arr = np.zeros(n_alpha)
    A2_arr = np.zeros(n_alpha)
    Cl_arr = np.zeros(n_alpha)
    Cm_arr = np.zeros(n_alpha)

    for i, (a_rad, a_deg) in enumerate(zip(alpha_rad, alpha_deg)):
        coeff = compute_fourier_coefficients(camber_func, a_rad, chord, n_terms=20, n_quad=n_quad)
        A0 = coeff['A0']
        An = coeff['An']

        A0_arr[i] = A0
        A1_arr[i] = An[0]   # A1
        A2_arr[i] = An[1]   # A2
        Cl_arr[i] = compute_Cl(A0, An[0])
        Cm_arr[i] = compute_Cm_quarter_chord(An[0], An[1])

    return {
        'alpha_deg': alpha_deg,
        'A0': A0_arr,
        'A1': A1_arr,
        'A2': A2_arr,
        'Cl': Cl_arr,
        'Cm': Cm_arr,
    }

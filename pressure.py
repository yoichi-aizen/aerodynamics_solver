"""
pressure.py
===========
Purpose : Compute the chordwise pressure-coefficient (Cp) distribution on the
          upper and lower surfaces of a thin airfoil using linearised thin
          airfoil theory.

Inputs  : Fourier coefficients A0, An; freestream velocity V∞; chord length.

Output  : Arrays of x-positions and Cp values for both surfaces.

Theory (Linearised, incompressible thin airfoil theory)
-------------------------------------------------------
In thin airfoil theory the vortex sheet lies on the camber line (z ≈ 0).
The tangential velocity just above and below the sheet differs by γ(x):

    V_upper = V∞ cos α + γ(x)/2   ≈   V∞ + γ(x)/2   (small α)
    V_lower = V∞ cos α − γ(x)/2   ≈   V∞ − γ(x)/2

Applying Bernoulli (linearised for small perturbations  u' = γ/2 << V∞):
    Cp = 1 − (V/V∞)²  ≈  −2 u'/V∞

Hence:
    Cp_upper(x) = −γ(x) / V∞      (suction on upper surface → negative Cp)
    Cp_lower(x) = +γ(x) / V∞      (pressure on lower surface → positive Cp)

Pressure difference:
    ΔCp = Cp_lower − Cp_upper = 2 γ(x) / V∞

Adverse pressure gradient (Bonus)
---------------------------------
The adverse pressure gradient (APG) on the upper surface is:
    dCp_upper/dx  (positive means rising pressure → potential separation)

We identify the region of maximum APG magnitude to assess separation risk.

Assumptions
-----------
* Linearised potential flow — valid only for small α and attached flow.
* Thin airfoil: camber and thickness ≪ chord.
* Pressure coefficient is approximate near the leading edge where the
  leading-edge singularity in γ is present.
"""

import numpy as np
from aerodynamics import compute_gamma_distribution


def compute_pressure_distribution(A0: float, An: np.ndarray,
                                   V_inf: float,
                                   alpha_rad: float,
                                   chord: float = 1.0,
                                   n_points: int = 400):
    """
    Compute Cp on upper and lower surfaces using linearised thin airfoil theory.

    Parameters
    ----------
    A0        : float      Zeroth Fourier coefficient
    An        : ndarray    Higher Fourier coefficients (A1 … AN)
    V_inf     : float      Freestream velocity [m/s]
    alpha_rad : float      Angle of attack [rad] (for reference in output only)
    chord     : float      Chord length [m]
    n_points  : int        Number of chordwise evaluation points

    Returns
    -------
    x_arr      : ndarray (n_points,)   Chord positions  x/c ∈ (0, 1)
    Cp_upper   : ndarray (n_points,)   Cp on upper surface (typically ≤ 0)
    Cp_lower   : ndarray (n_points,)   Cp on lower surface (typically ≥ 0)
    Cp_diff    : ndarray (n_points,)   ΔCp = Cp_lower − Cp_upper
    apg_x      : ndarray               x positions where APG > 0 on upper surface
    apg_vals   : ndarray               dCp_upper/dx values at those positions
    """
    # Get circulation distribution γ(θ) along the camber line
    theta_arr, x_arr, gamma_arr = compute_gamma_distribution(
        A0, An, V_inf, chord, n_points
    )

    # Pressure coefficients (linearised)
    Cp_upper = -gamma_arr / V_inf    # suction side
    Cp_lower = +gamma_arr / V_inf    # pressure side

    Cp_diff = Cp_lower - Cp_upper    # = 2γ/V∞

    # Adverse pressure gradient on upper surface:  dCp/dx > 0
    dCp_dx = np.gradient(Cp_upper, x_arr)
    apg_mask = dCp_dx > 0
    apg_x = x_arr[apg_mask]
    apg_vals = dCp_dx[apg_mask]

    return x_arr, Cp_upper, Cp_lower, Cp_diff, apg_x, apg_vals


def find_minimum_apg_camber(camber_expr_list: list,
                             alpha_rad: float,
                             V_inf: float = 1.0,
                             chord: float = 1.0):
    """
    Compare several camber line designs and rank by adverse pressure gradient.

    Evaluates the RMS adverse pressure gradient magnitude on the upper surface
    for each design at the given angle of attack.  Lower RMS → less tendency
    to separate.

    Parameters
    ----------
    camber_expr_list : list of (name, camber_func) tuples
    alpha_rad        : float   Angle of attack [rad]
    V_inf            : float   Freestream velocity [m/s]
    chord            : float   Chord length [m]

    Returns
    -------
    list of (name, rms_apg, Cl) sorted by ascending rms_apg
    """
    from aerodynamics import compute_fourier_coefficients, compute_Cl

    results = []
    for name, camber_func in camber_expr_list:
        coeff = compute_fourier_coefficients(camber_func, alpha_rad, chord)
        A0, An = coeff['A0'], coeff['An']
        Cl = compute_Cl(A0, An[0])

        _, _, _, _, apg_x, apg_vals = compute_pressure_distribution(
            A0, An, V_inf, alpha_rad, chord
        )
        rms_apg = float(np.sqrt(np.mean(apg_vals**2))) if len(apg_vals) > 0 else 0.0
        results.append((name, rms_apg, Cl))

    results.sort(key=lambda r: r[1])   # sort by RMS APG (ascending = better)
    return results

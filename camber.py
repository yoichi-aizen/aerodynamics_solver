"""
camber.py
=========
Purpose : Generate camber line functions for NACA 4-digit airfoils,
          user-supplied mathematical expressions, and CSV data files.

Inputs  : NACA string (e.g. '2412'), expression string (e.g. '0.06*sin(pi*x)'),
          or CSV file path containing columns [x, z].

Outputs : A callable  z = f(x)  valid for  x in [0, 1]  (normalised chord).

Assumptions
-----------
* Leading edge at x = 0, trailing edge at x = 1 (chord = 1 by default).
* NACA 4-digit formulation only (not 5-digit or 6-series).
* Expression strings may use standard numpy maths (sin, cos, pi, sqrt …).
* CSV interpolation uses a natural cubic spline; data must span [0, 1].
"""

import numpy as np
from scipy.interpolate import CubicSpline
import csv
import os


# ---------------------------------------------------------------------------
# NACA 4-digit camber line
# ---------------------------------------------------------------------------

def naca4_camber(naca_str: str):
    """
    Return the camber-line function for a NACA 4-digit airfoil.

    Parameters
    ----------
    naca_str : str
        Four-character NACA number, e.g. '2412'.
        Digit 1 → max camber  m = digit / 100
        Digit 2 → chordwise position of max camber  p = digit / 10
        Digits 3-4 → max thickness (ignored here – thin airfoil uses camber only)

    Returns
    -------
    callable  z = f(x),  x ∈ [0, 1]
    """
    if len(naca_str) != 4 or not naca_str.isdigit():
        raise ValueError(f"'{naca_str}' is not a valid NACA 4-digit code.")

    m = int(naca_str[0]) / 100.0   # maximum camber fraction
    p = int(naca_str[1]) / 10.0    # position of max camber fraction

    if m == 0.0 or p == 0.0:       # symmetric airfoil
        def camber_func(x):
            x = np.asarray(x, dtype=float)
            return np.zeros_like(x) if x.ndim > 0 else 0.0
    else:
        def camber_func(x):
            x = np.asarray(x, dtype=float)
            scalar_in = x.ndim == 0
            x = np.atleast_1d(x)
            z = np.where(
                x < p,
                (m / p**2) * (2.0 * p * x - x**2),
                (m / (1.0 - p)**2) * (1.0 - 2.0 * p + 2.0 * p * x - x**2)
            )
            return float(z[0]) if scalar_in else z

    return camber_func


# ---------------------------------------------------------------------------
# Custom camber line from a math expression
# ---------------------------------------------------------------------------

# Safe namespace for evaluating user expressions
_SAFE_NS = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'exp': np.exp, 'log': np.log, 'log10': np.log10,
    'sqrt': np.sqrt, 'abs': np.abs,
    'pi': np.pi, 'e': np.e,
    'np': np,
}


def custom_camber_from_expr(expr_str: str):
    """
    Return a camber-line function defined by a mathematical expression.

    The expression must be a function of the variable  x  (chord position 0–1).
    Any numpy function is available (sin, cos, pi, sqrt, exp, …).

    Example
    -------
    '0.06 * sin(pi * x)'
    '0.04 * x * (1 - x)'
    '0.08 * (x**2) * (1 - x)'

    Parameters
    ----------
    expr_str : str   The math expression as a string.

    Returns
    -------
    callable  z = f(x)

    Raises
    ------
    ValueError  if the expression cannot be evaluated.
    """
    # Validate expression before returning the function
    try:
        ns = dict(_SAFE_NS)
        ns['x'] = np.array([0.0, 0.5, 1.0])
        result = eval(expr_str, {"__builtins__": {}}, ns)
        _ = float(np.asarray(result).mean())   # ensure it returns a number
    except Exception as exc:
        raise ValueError(f"Invalid expression '{expr_str}': {exc}") from exc

    def camber_func(x):
        ns = dict(_SAFE_NS)
        ns['x'] = x
        return eval(expr_str, {"__builtins__": {}}, ns)

    camber_func.__doc__ = f"Custom camber: z = {expr_str}"
    return camber_func


def validate_expr(expr_str: str) -> str:
    """Return '' if expression is valid, or an error message."""
    try:
        custom_camber_from_expr(expr_str)
        return ''
    except ValueError as e:
        return str(e)


# ---------------------------------------------------------------------------
# Custom camber line from CSV
# ---------------------------------------------------------------------------

def custom_camber_from_csv(filepath: str):
    """
    Return a camber-line function interpolated from a CSV data file.

    The CSV must contain at least two columns: x (chord position) and z (camber).
    A header row is optional (if the first cell is non-numeric it is skipped).

    Parameters
    ----------
    filepath : str   Path to the CSV file.

    Returns
    -------
    callable  z = f(x)  (natural cubic spline, extrapolation clamped to endpoints)

    Raises
    ------
    FileNotFoundError, ValueError
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: '{filepath}'")

    x_data, z_data = [], []
    with open(filepath, 'r', newline='') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                x_data.append(float(row[0]))
                z_data.append(float(row[1]))
            except ValueError:
                pass  # skip header or non-numeric rows

    if len(x_data) < 4:
        raise ValueError("CSV must contain at least 4 valid (x, z) rows for spline fitting.")

    x_arr = np.array(x_data)
    z_arr = np.array(z_data)
    order = np.argsort(x_arr)          # sort by x
    x_arr, z_arr = x_arr[order], z_arr[order]

    spline = CubicSpline(x_arr, z_arr, bc_type='natural')

    def camber_func(x):
        x_c = np.clip(x, x_arr[0], x_arr[-1])
        return spline(x_c)

    return camber_func


# ---------------------------------------------------------------------------
# Utility: sample camber line
# ---------------------------------------------------------------------------

def get_camber_points(camber_func, n_points: int = 300):
    """
    Sample the camber line and return coordinate arrays.

    Parameters
    ----------
    camber_func : callable    z = f(x) camber line function
    n_points    : int         Number of sample points

    Returns
    -------
    x_arr : ndarray  shape (n_points,)
    z_arr : ndarray  shape (n_points,)
    """
    x_arr = np.linspace(0.0, 1.0, n_points)
    # Call element-wise to support scalar-only implementations
    try:
        z_arr = np.asarray(camber_func(x_arr), dtype=float)
    except Exception:
        z_arr = np.array([float(camber_func(xi)) for xi in x_arr])
    return x_arr, z_arr

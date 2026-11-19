===============================================================
  Thin Airfoil Theory Tool — Para Glider Design (AE 244)
===============================================================

REQUIREMENTS
------------
Python 3.9 or later with the following packages:
    pip install numpy scipy matplotlib

RUNNING THE TOOL
----------------
Navigate to this folder and run:
    python main.py

A window will open with four tabs in the left panel and
a live matplotlib canvas on the right.

QUICK START
-----------
1. NACA Tab → type a 4-digit NACA number (e.g. 2412) → click "Load"
2. Adjust V∞ and α sliders
3. Click any plot button to generate results on the right canvas
4. Use the matplotlib toolbar (zoom, pan, save) on the canvas

LOADING CUSTOM AIRFOILS
-----------------------
Expression : go to "Custom Airfoil" tab, type z = f(x) in terms of x
             e.g.  0.08*sin(pi*x)   or   0.12*x*(1-x)
             then click "Load Expression as Active Airfoil"

CSV file   : in the same tab click "Browse CSV …"
             CSV format: two columns, x (0–1) and z
             Example:
               x,z
               0.0,0.000
               0.25,0.045
               0.5,0.060
               0.75,0.040
               1.0,0.000

CFD COMPARISON
--------------
In the NACA tab, click "Load CSV" next to "CFD Comparison Data".
CSV must have columns: alpha (or aoa), Cl, Cm  (header required)
Example:
  alpha,Cl,Cm
  -3,0.05,-0.04
   0,0.32,-0.04
   3,0.60,-0.04
   6,0.87,-0.04

3 CUSTOM DESIGNS
----------------
In the "3 Designs" tab, enter names and expressions for each design,
click "Load All 3 Designs", then use the comparison plot buttons.
A NACA airfoil must be loaded for NACA to appear in the comparison.

BONUS: PRESSURE DISTRIBUTION
-----------------------------
Go to the "Bonus: Pressure" tab and click "Plot Pressure Distribution".
The tool shows Cp_upper, Cp_lower, ΔCp, and highlights the adverse
pressure gradient (APG) region on the upper surface.
"Rank Designs by Min APG" compares all loaded airfoils.

FILES
-----
  main.py           GUI entry point (run this)
  camber.py         Camber line generation (NACA, expression, CSV)
  aerodynamics.py   Thin airfoil theory — Fourier coefficients, Cl, Cm
  velocity_field.py Velocity field (Biot-Savart), line-integral circulation
  pressure.py       Pressure distribution (Bonus)
  README.txt        This file

SAVING PLOTS
------------
Use the floppy-disk icon in the matplotlib toolbar to save any plot as
PNG, PDF, SVG, or EPS.

KNOWN LIMITATIONS
-----------------
* Thin airfoil theory is valid for small α (<~10°) and attached flow.
* Velocity field computation can take 5-20 seconds depending on grid size.
* Line integral computation can take ~10-20 seconds.
* Leading-edge singularity in γ(θ) causes γ → ∞ at x/c → 0 (normal for TAT).

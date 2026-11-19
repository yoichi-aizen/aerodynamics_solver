"""
main.py
=======
Purpose : Main entry-point for the Thin Airfoil Theory Analysis Tool.
          Provides a tkinter GUI with embedded matplotlib figures.

Usage   : python main.py

Inputs  (via GUI)
-----------------
* Airfoil type  : NACA 4-digit | Custom expression | Custom CSV file
* Flow params   : Freestream velocity V∞, Angle of attack α
* Optional CFD  : CSV with columns [alpha_deg, Cl, Cm] for overlay plots
* Custom designs: Up to 3 user-defined airfoils for comparison

Outputs (via GUI plots)
-----------------------
* Camber line  z vs x
* Camber slope dz/dx vs x
* Fourier coefficients A0, A1, A2 vs α
* Cl vs α  (with optional CFD overlay)
* Cm vs α  (with optional CFD overlay)
* γ(θ) circulation distribution at current α
* Bound circulation Γ (numerical integral)
* Circulation via velocity line-integral
* Velocity field (4c × 3c domain)
* Pressure distribution Cp_upper, Cp_lower (Bonus)
* Camber / Cl / Cm / velocity-field comparison of 3 custom designs

Assumptions
-----------
* Thin airfoil theory — see individual module preambles for details.
* CFD CSV must have header row with columns: alpha, Cl, Cm
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# Add the tool directory to path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camber import (naca4_camber, custom_camber_from_expr,
                    custom_camber_from_csv, get_camber_points, validate_expr)
from aerodynamics import (compute_slope, compute_fourier_coefficients,
                           compute_Cl, compute_Cm_quarter_chord,
                           compute_gamma_distribution,
                           compute_bound_circulation, sweep_alpha)
from velocity_field import (compute_velocity_field,
                              compute_circulation_line_integral,
                              get_vortex_elements)
from pressure import compute_pressure_distribution


# ---------------------------------------------------------------------------
# Colour palette (consistent across all plots)
# ---------------------------------------------------------------------------
COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
          '#00BCD4', '#795548', '#607D8B']
BG = '#1e1e2e'      # dark background
FG = '#cdd6f4'      # foreground text
PANEL_BG = '#313244'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': '#2a2a3e',
    'axes.edgecolor': '#585b70', 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG, 'text.color': FG,
    'grid.color': '#45475a', 'grid.alpha': 0.5,
    'legend.facecolor': '#313244', 'legend.edgecolor': '#585b70',
    'legend.labelcolor': FG,
    'font.size': 9, 'axes.titlesize': 10,
})


# ============================================================================
# Helper: load CFD CSV
# ============================================================================

def load_cfd_csv(filepath: str):
    """Load CFD data CSV. Returns dict with arrays alpha_deg, Cl, Cm or None."""
    try:
        data = np.genfromtxt(filepath, delimiter=',', names=True,
                              deletechars='', dtype=None, encoding='utf-8')
        names_lower = [n.lower().strip() for n in data.dtype.names]
        def get_col(keywords):
            for k in keywords:
                for n, nl in zip(data.dtype.names, names_lower):
                    if k in nl:
                        return data[n].astype(float)
            return None
        alpha = get_col(['alpha', 'aoa', 'angle'])
        cl    = get_col(['cl', 'lift'])
        cm    = get_col(['cm', 'moment'])
        if alpha is None or cl is None:
            raise ValueError("Could not find alpha and Cl columns.")
        return {'alpha_deg': alpha, 'Cl': cl, 'Cm': cm}
    except Exception as e:
        messagebox.showerror("CFD Load Error", str(e))
        return None


# ============================================================================
# Main Application
# ============================================================================

class AirfoilApp:
    """
    Main tkinter application for Thin Airfoil Theory analysis.

    Layout
    ------
    Left panel  : ttk.Notebook with control tabs
    Right panel : matplotlib figure canvas + navigation toolbar
    Bottom bar  : status messages and progress indicator
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Thin Airfoil Theory Tool — Para Glider Design (AE 244)")
        self.root.configure(bg=BG)
        try:
            self.root.state('zoomed')
        except Exception:
            self.root.geometry('1400x800')

        # ---- State variables ----
        self.camber_func = None      # current airfoil camber function
        self.camber_label = tk.StringVar(value="No airfoil loaded")
        self.cfd_data = None         # loaded CFD comparison data

        # Custom design slots (3 designs)
        self.design_funcs = [None, None, None]
        self.design_names = ['Design A', 'Design B', 'Design C']

        # Flow parameters
        self.v_inf_var   = tk.DoubleVar(value=20.0)   # m/s
        self.alpha_var   = tk.DoubleVar(value=3.0)    # degrees
        self.alpha_min_v = tk.DoubleVar(value=-3.0)
        self.alpha_max_v = tk.DoubleVar(value=12.0)
        self.n_alpha_v   = tk.IntVar(value=50)

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # Main horizontal paned window
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                bg=BG, sashwidth=4, sashrelief='flat')
        paned.pack(fill='both', expand=True)

        # ---- LEFT PANEL ----
        left_frame = tk.Frame(paned, bg=PANEL_BG, width=340)
        left_frame.pack_propagate(False)
        paned.add(left_frame, minsize=300)

        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill='both', expand=True, padx=4, pady=4)

        self._build_naca_tab()
        self._build_custom_tab()
        self._build_designs_tab()
        self._build_bonus_tab()

        # ---- RIGHT PANEL ----
        right_frame = tk.Frame(paned, bg=BG)
        paned.add(right_frame, minsize=700)

        # matplotlib figure
        self.fig = Figure(figsize=(10, 7), facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar_frame = tk.Frame(right_frame, bg=BG)
        toolbar_frame.pack(fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # ---- STATUS BAR ----
        self.status_var = tk.StringVar(value="Ready. Load an airfoil to begin.")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                               bg='#181825', fg='#a6e3a1',
                               anchor='w', padx=8, font=('Consolas', 9))
        status_bar.pack(side='bottom', fill='x')

    # -----------------------------------------------------------------------
    # Tab 1: NACA Airfoil
    # -----------------------------------------------------------------------

    def _build_naca_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" NACA Airfoil ")

        self._section(tab, "Airfoil Definition")

        row = tk.Frame(tab, bg=PANEL_BG)
        row.pack(fill='x', padx=6, pady=2)
        tk.Label(row, text="NACA Number:", bg=PANEL_BG, fg=FG,
                  font=('Segoe UI', 9)).pack(side='left')
        self.naca_entry = tk.Entry(row, width=8, bg='#2a2a3e', fg=FG,
                                    insertbackground=FG, font=('Consolas', 10))
        self.naca_entry.insert(0, "2412")
        self.naca_entry.pack(side='left', padx=6)
        tk.Button(row, text="Load", command=self._load_naca,
                   bg='#89b4fa', fg='#1e1e2e', font=('Segoe UI', 9, 'bold'),
                   relief='flat', padx=8).pack(side='left')

        # Loaded airfoil label
        tk.Label(tab, textvariable=self.camber_label, bg=PANEL_BG,
                  fg='#a6e3a1', font=('Consolas', 8),
                  wraplength=300, justify='left').pack(fill='x', padx=8, pady=2)

        self._section(tab, "CFD Comparison Data (optional)")
        row2 = tk.Frame(tab, bg=PANEL_BG)
        row2.pack(fill='x', padx=6, pady=2)
        self.cfd_label = tk.StringVar(value="No CFD data loaded")
        tk.Label(row2, textvariable=self.cfd_label, bg=PANEL_BG,
                  fg='#f38ba8', font=('Consolas', 8)).pack(side='left', padx=4)
        tk.Button(row2, text="Load CSV", command=self._load_cfd,
                   bg='#f38ba8', fg='#1e1e2e', font=('Segoe UI', 8),
                   relief='flat', padx=6).pack(side='right')

        self._section(tab, "Flow Parameters")
        self._slider_row(tab, "V∞ (m/s)", self.v_inf_var, 1, 100, 20)
        self._slider_row(tab, "α (deg)", self.alpha_var, -10, 20, 3)

        self._section(tab, "Alpha Sweep Range (for curves)")
        self._slider_row(tab, "α min (deg)", self.alpha_min_v, -10, 0, -3)
        self._slider_row(tab, "α max (deg)", self.alpha_max_v, 5, 25, 12)
        self._num_row(tab, "# points", self.n_alpha_v)

        self._section(tab, "Plots")
        buttons = [
            ("Camber Line  z vs x",          self._plot_camber),
            ("Camber Slope  dz/dx vs x",      self._plot_slope),
            ("Fourier Coeffs A0 A1 A2 vs α",  self._plot_fourier),
            ("Cl vs α  (+ CFD overlay)",       self._plot_Cl),
            ("Cm vs α  (+ CFD overlay)",       self._plot_Cm),
            ("γ(θ) Distribution at α",         self._plot_gamma),
            ("Velocity Field (4c × 3c)",       self._plot_vfield),
        ]
        for label, cmd in buttons:
            tk.Button(tab, text=label, command=cmd,
                       bg='#313244', fg=FG, relief='flat',
                       font=('Segoe UI', 9), anchor='w',
                       activebackground='#45475a', activeforeground=FG,
                       pady=4, padx=8).pack(fill='x', padx=6, pady=1)

        self._section(tab, "Circulation (at current α)")
        tk.Button(tab, text="Compute Bound Γ (Distribution Integral)",
                   command=self._compute_bound_gamma,
                   bg='#a6e3a1', fg='#1e1e2e', relief='flat',
                   font=('Segoe UI', 9, 'bold'), pady=4).pack(fill='x', padx=6, pady=1)
        tk.Button(tab, text="Compute Γ (Velocity Line Integral)",
                   command=self._compute_line_integral,
                   bg='#cba6f7', fg='#1e1e2e', relief='flat',
                   font=('Segoe UI', 9, 'bold'), pady=4).pack(fill='x', padx=6, pady=1)

        self.circulation_text = tk.StringVar(value="")
        tk.Label(tab, textvariable=self.circulation_text, bg=PANEL_BG,
                  fg='#fab387', font=('Consolas', 9),
                  justify='left', wraplength=300).pack(fill='x', padx=8, pady=4)

    # -----------------------------------------------------------------------
    # Tab 2: Custom Single Airfoil
    # -----------------------------------------------------------------------

    def _build_custom_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Custom Airfoil ")

        self._section(tab, "Define Custom Camber  z = f(x)")
        tk.Label(tab, text="Expression  (use x, sin, cos, pi, sqrt, exp …):",
                  bg=PANEL_BG, fg=FG, font=('Segoe UI', 8)).pack(anchor='w', padx=8)

        self.expr_var = tk.StringVar(value="0.06*sin(pi*x)")
        expr_entry = tk.Entry(tab, textvariable=self.expr_var, width=38,
                               bg='#2a2a3e', fg='#a6e3a1',
                               insertbackground=FG, font=('Consolas', 9))
        expr_entry.pack(fill='x', padx=8, pady=2)

        tk.Button(tab, text="Load Expression as Active Airfoil",
                   command=self._load_expr,
                   bg='#89b4fa', fg='#1e1e2e', relief='flat',
                   font=('Segoe UI', 9, 'bold'), pady=4).pack(fill='x', padx=8, pady=2)

        self._section(tab, "— or upload a CSV file —")
        self.csv_label = tk.StringVar(value="No CSV selected")
        tk.Label(tab, textvariable=self.csv_label, bg=PANEL_BG,
                  fg='#f9e2af', font=('Consolas', 8),
                  wraplength=300, justify='left').pack(fill='x', padx=8)
        tk.Button(tab, text="Browse CSV …", command=self._load_custom_csv,
                   bg='#f9e2af', fg='#1e1e2e', relief='flat',
                   font=('Segoe UI', 9), pady=3).pack(fill='x', padx=8, pady=2)

        self._section(tab, "Expression Examples")
        examples = [
            "NACA-like:       0.06*(2*0.4*x - x**2)/0.4**2  if x<0.4",
            "Sine arc:        0.08*sin(pi*x)",
            "Quadratic:       0.12*x*(1-x)",
            "Cubic:           0.15*x**2*(1-x)",
            "Skewed sine:     0.1*sin(pi*x**0.7)",
        ]
        for ex in examples:
            tk.Label(tab, text=ex, bg=PANEL_BG, fg='#6c7086',
                      font=('Consolas', 7), anchor='w').pack(fill='x', padx=10)

    # -----------------------------------------------------------------------
    # Tab 3: 3 Custom Designs Comparison
    # -----------------------------------------------------------------------

    def _build_designs_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" 3 Designs ")

        self._section(tab, "Define 3 Custom Airfoil Designs")
        tk.Label(tab, text="Each design: a math expression  z = f(x)",
                  bg=PANEL_BG, fg=FG, font=('Segoe UI', 8)).pack(anchor='w', padx=8, pady=2)

        self.design_vars = []
        self.design_name_vars = []
        design_defaults = [
            ("Design A", "0.08*sin(pi*x)"),
            ("Design B", "0.12*x*(1-x)"),
            ("Design C", "0.1*x**2*(1-x)"),
        ]
        for i, (dname, dexpr) in enumerate(design_defaults):
            f = tk.LabelFrame(tab, text=f" Design {chr(65+i)} ",
                               bg=PANEL_BG, fg='#89b4fa',
                               font=('Segoe UI', 9), padx=4, pady=4)
            f.pack(fill='x', padx=6, pady=3)

            nv = tk.StringVar(value=dname)
            self.design_name_vars.append(nv)
            row = tk.Frame(f, bg=PANEL_BG)
            row.pack(fill='x')
            tk.Label(row, text="Name:", bg=PANEL_BG, fg=FG,
                      font=('Segoe UI', 8)).pack(side='left')
            tk.Entry(row, textvariable=nv, width=14, bg='#2a2a3e', fg=FG,
                      insertbackground=FG, font=('Consolas', 8)).pack(side='left', padx=4)

            ev = tk.StringVar(value=dexpr)
            self.design_vars.append(ev)
            tk.Entry(f, textvariable=ev, width=36, bg='#2a2a3e',
                      fg='#a6e3a1', insertbackground=FG,
                      font=('Consolas', 8)).pack(fill='x', pady=2)

        tk.Button(tab, text="Load All 3 Designs",
                   command=self._load_designs,
                   bg='#a6e3a1', fg='#1e1e2e', relief='flat',
                   font=('Segoe UI', 9, 'bold'), pady=4).pack(fill='x', padx=8, pady=4)

        self._section(tab, "Comparison Plots (requires NACA airfoil loaded)")
        comp_buttons = [
            ("Camber Lines Comparison",        self._plot_compare_camber),
            ("Cl vs α Comparison",             self._plot_compare_Cl),
            ("Cm vs α Comparison",             self._plot_compare_Cm),
            ("Velocity Fields (2×2 grid)",     self._plot_compare_vfields),
        ]
        for label, cmd in comp_buttons:
            tk.Button(tab, text=label, command=cmd,
                       bg='#313244', fg=FG, relief='flat',
                       font=('Segoe UI', 9), anchor='w',
                       activebackground='#45475a',
                       pady=4, padx=8).pack(fill='x', padx=6, pady=1)

    # -----------------------------------------------------------------------
    # Tab 4: Bonus — Pressure Distribution
    # -----------------------------------------------------------------------

    def _build_bonus_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Bonus: Pressure ")

        self._section(tab, "Pressure Distribution  Cp(x)")
        tk.Label(tab,
                  text=("Computes Cp on upper and lower surfaces\n"
                         "using linearised thin airfoil theory.\n\n"
                         "Cp_upper = −γ(x)/V∞\n"
                         "Cp_lower = +γ(x)/V∞\n\n"
                         "Adverse pressure gradient (dCp/dx > 0)\n"
                         "is highlighted — indicates separation risk."),
                  bg=PANEL_BG, fg=FG, font=('Consolas', 8),
                  justify='left').pack(fill='x', padx=10, pady=6)

        tk.Button(tab, text="Plot Pressure Distribution",
                   command=self._plot_pressure,
                   bg='#fab387', fg='#1e1e2e', relief='flat',
                   font=('Segoe UI', 10, 'bold'), pady=6).pack(fill='x', padx=8, pady=4)

        self._section(tab, "APG Minimiser")
        tk.Label(tab,
                  text=("Compares all loaded designs\n"
                         "and ranks by RMS adverse pressure\n"
                         "gradient on the upper surface.\n"
                         "Lower RMS → less separation tendency."),
                  bg=PANEL_BG, fg=FG, font=('Consolas', 8),
                  justify='left').pack(fill='x', padx=10, pady=4)

        tk.Button(tab, text="Rank Designs by Min APG",
                   command=self._rank_apg,
                   bg='#89dceb', fg='#1e1e2e', relief='flat',
                   font=('Segoe UI', 9, 'bold'), pady=4).pack(fill='x', padx=8, pady=4)

        self.apg_text = tk.StringVar(value="")
        tk.Label(tab, textvariable=self.apg_text, bg=PANEL_BG,
                  fg='#f9e2af', font=('Consolas', 8),
                  justify='left', wraplength=300).pack(fill='x', padx=8, pady=4)

    # -----------------------------------------------------------------------
    # Widget helpers
    # -----------------------------------------------------------------------

    def _section(self, parent, text: str):
        tk.Label(parent, text=f"  {text}",
                  bg='#45475a', fg='#cba6f7',
                  font=('Segoe UI', 8, 'bold'),
                  anchor='w').pack(fill='x', padx=0, pady=(6, 1))

    def _slider_row(self, parent, label: str, var, from_, to_, default):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill='x', padx=6, pady=1)
        tk.Label(row, text=f"{label}:", bg=PANEL_BG, fg=FG,
                  font=('Segoe UI', 8), width=14, anchor='w').pack(side='left')
        val_lbl = tk.Label(row, text=f"{default:.1f}", bg=PANEL_BG,
                            fg='#89b4fa', font=('Consolas', 8), width=6)
        val_lbl.pack(side='right')
        sl = ttk.Scale(row, from_=from_, to=to_, variable=var,
                        orient='horizontal', length=130)
        sl.pack(side='left', padx=4)
        def update_lbl(*_):
            val_lbl.config(text=f"{var.get():.1f}")
        var.trace_add('write', update_lbl)

    def _num_row(self, parent, label: str, var):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill='x', padx=6, pady=1)
        tk.Label(row, text=f"{label}:", bg=PANEL_BG, fg=FG,
                  font=('Segoe UI', 8), width=14, anchor='w').pack(side='left')
        tk.Entry(row, textvariable=var, width=6, bg='#2a2a3e', fg=FG,
                  insertbackground=FG, font=('Consolas', 8)).pack(side='left', padx=4)

    # -----------------------------------------------------------------------
    # Airfoil Loading
    # -----------------------------------------------------------------------

    def _load_naca(self):
        naca = self.naca_entry.get().strip()
        try:
            self.camber_func = naca4_camber(naca)
            self.camber_label.set(f"NACA {naca} loaded ✓")
            self._set_status(f"NACA {naca} airfoil loaded successfully.")
        except ValueError as e:
            messagebox.showerror("NACA Error", str(e))

    def _load_expr(self):
        expr = self.expr_var.get().strip()
        err = validate_expr(expr)
        if err:
            messagebox.showerror("Expression Error", err)
            return
        self.camber_func = custom_camber_from_expr(expr)
        self.camber_label.set(f"Custom: z = {expr[:40]}…" if len(expr) > 40 else f"Custom: z = {expr}")
        self._set_status(f"Custom expression loaded: z = {expr}")

    def _load_custom_csv(self):
        path = filedialog.askopenfilename(
            title="Select camber CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.camber_func = custom_camber_from_csv(path)
            fname = os.path.basename(path)
            self.csv_label.set(f"Loaded: {fname}")
            self.camber_label.set(f"CSV: {fname} loaded ✓")
            self._set_status(f"Custom CSV camber loaded from: {fname}")
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))

    def _load_cfd(self):
        path = filedialog.askopenfilename(
            title="Select CFD data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        data = load_cfd_csv(path)
        if data:
            self.cfd_data = data
            fname = os.path.basename(path)
            self.cfd_label.set(f"CFD: {fname} ✓")
            self._set_status(f"CFD data loaded from: {fname} ({len(data['alpha_deg'])} points)")

    def _load_designs(self):
        loaded = 0
        for i in range(3):
            expr = self.design_vars[i].get().strip()
            name = self.design_name_vars[i].get().strip()
            self.design_names[i] = name
            err = validate_expr(expr)
            if err:
                messagebox.showerror(f"Design {chr(65+i)} Error", err)
                self.design_funcs[i] = None
            else:
                self.design_funcs[i] = custom_camber_from_expr(expr)
                loaded += 1
        self._set_status(f"{loaded}/3 custom designs loaded successfully.")

    # -----------------------------------------------------------------------
    # Guard: ensure airfoil is loaded
    # -----------------------------------------------------------------------

    def _require_airfoil(self) -> bool:
        if self.camber_func is None:
            messagebox.showwarning("No Airfoil",
                                    "Please load an airfoil first (NACA tab or Custom tab).")
            return False
        return True

    def _get_flow(self):
        """Return current (V_inf, alpha_rad, alpha_deg) from sliders."""
        V_inf     = self.v_inf_var.get()
        alpha_deg = self.alpha_var.get()
        alpha_rad = np.deg2rad(alpha_deg)
        return V_inf, alpha_rad, alpha_deg

    def _set_status(self, msg: str):
        self.status_var.set(f"  {msg}")
        self.root.update_idletasks()

    def _clear_fig(self):
        self.fig.clear()

    def _draw(self):
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    # -----------------------------------------------------------------------
    # Plot: Camber Line
    # -----------------------------------------------------------------------

    def _plot_camber(self):
        if not self._require_airfoil():
            return
        self._set_status("Plotting camber line …")
        x, z = get_camber_points(self.camber_func)

        self._clear_fig()
        ax = self.fig.add_subplot(111)
        ax.plot(x, z, color=COLORS[0], lw=2.5, label='Camber line')
        ax.axhline(0, color='#585b70', lw=0.8, linestyle='--')
        ax.set_xlabel("x/c   (normalised chord)")
        ax.set_ylabel("z/c   (normalised camber)")
        ax.set_title("Airfoil Camber Line   z vs x")
        ax.set_xlim(0, 1)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True)
        self._draw()
        self._set_status("Camber line plotted.")

    # -----------------------------------------------------------------------
    # Plot: Camber Slope
    # -----------------------------------------------------------------------

    def _plot_slope(self):
        if not self._require_airfoil():
            return
        self._set_status("Computing camber slope …")
        x_vals = np.linspace(0.005, 0.995, 400)
        dzdx = compute_slope(self.camber_func, x_vals)

        self._clear_fig()
        ax = self.fig.add_subplot(111)
        ax.plot(x_vals, dzdx, color=COLORS[1], lw=2.5, label='dz/dx')
        ax.axhline(0, color='#585b70', lw=0.8, linestyle='--')
        ax.set_xlabel("x/c")
        ax.set_ylabel("dz/dx")
        ax.set_title("Camber Line Slope   dz/dx vs x/c")
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True)
        self._draw()
        self._set_status("Camber slope plotted.")

    # -----------------------------------------------------------------------
    # Plot: Fourier Coefficients vs alpha
    # -----------------------------------------------------------------------

    def _plot_fourier(self):
        if not self._require_airfoil():
            return
        self._set_status("Computing Fourier coefficients over alpha sweep …")
        sw = sweep_alpha(self.camber_func,
                          self.alpha_min_v.get(), self.alpha_max_v.get(),
                          self.n_alpha_v.get())

        self._clear_fig()
        ax = self.fig.add_subplot(111)
        ax.plot(sw['alpha_deg'], sw['A0'], color=COLORS[0], lw=2, marker='o',
                 markersize=3, label='A₀')
        ax.plot(sw['alpha_deg'], sw['A1'], color=COLORS[1], lw=2, marker='s',
                 markersize=3, label='A₁')
        ax.plot(sw['alpha_deg'], sw['A2'], color=COLORS[2], lw=2, marker='^',
                 markersize=3, label='A₂')
        ax.axhline(0, color='#585b70', lw=0.7, linestyle='--')
        ax.set_xlabel("Angle of Attack α (deg)")
        ax.set_ylabel("Fourier Coefficient")
        ax.set_title("Fourier Coefficients A₀, A₁, A₂ vs Angle of Attack")
        ax.legend()
        ax.grid(True)
        self._draw()
        self._set_status("Fourier coefficient sweep complete.")

    # -----------------------------------------------------------------------
    # Plot: Cl vs alpha
    # -----------------------------------------------------------------------

    def _plot_Cl(self):
        if not self._require_airfoil():
            return
        self._set_status("Computing Cl curve …")
        sw = sweep_alpha(self.camber_func,
                          self.alpha_min_v.get(), self.alpha_max_v.get(),
                          self.n_alpha_v.get())

        self._clear_fig()
        ax = self.fig.add_subplot(111)
        ax.plot(sw['alpha_deg'], sw['Cl'], color=COLORS[0], lw=2.5,
                 label='TAT (This tool)')

        if self.cfd_data:
            ax.plot(self.cfd_data['alpha_deg'], self.cfd_data['Cl'],
                     'o--', color=COLORS[3], lw=1.8, markersize=5,
                     label='CFD (Ansys)')

        ax.axhline(0, color='#585b70', lw=0.7, linestyle='--')
        ax.axvline(0, color='#585b70', lw=0.7, linestyle='--')
        ax.set_xlabel("Angle of Attack α (deg)")
        ax.set_ylabel("Section Lift Coefficient  Cl")
        ax.set_title("Lift Curve   Cl vs α")
        ax.legend()
        ax.grid(True)
        self._draw()
        self._set_status("Cl curve plotted.")

    # -----------------------------------------------------------------------
    # Plot: Cm vs alpha
    # -----------------------------------------------------------------------

    def _plot_Cm(self):
        if not self._require_airfoil():
            return
        self._set_status("Computing Cm curve …")
        sw = sweep_alpha(self.camber_func,
                          self.alpha_min_v.get(), self.alpha_max_v.get(),
                          self.n_alpha_v.get())

        self._clear_fig()
        ax = self.fig.add_subplot(111)
        ax.plot(sw['alpha_deg'], sw['Cm'], color=COLORS[1], lw=2.5,
                 label='TAT Cm_c/4 (This tool)')

        if self.cfd_data and self.cfd_data.get('Cm') is not None:
            ax.plot(self.cfd_data['alpha_deg'], self.cfd_data['Cm'],
                     'o--', color=COLORS[3], lw=1.8, markersize=5,
                     label='CFD (Ansys)')

        ax.axhline(0, color='#585b70', lw=0.7, linestyle='--')
        ax.set_xlabel("Angle of Attack α (deg)")
        ax.set_ylabel("Moment Coefficient  Cm (about c/4)")
        ax.set_title("Moment Curve   Cm vs α   (about quarter-chord)")
        ax.legend()
        ax.grid(True)
        self._draw()
        self._set_status("Cm curve plotted.")

    # -----------------------------------------------------------------------
    # Plot: Circulation distribution γ(θ)
    # -----------------------------------------------------------------------

    def _plot_gamma(self):
        if not self._require_airfoil():
            return
        V_inf, alpha_rad, alpha_deg = self._get_flow()
        self._set_status("Computing circulation distribution …")

        coeff = compute_fourier_coefficients(self.camber_func, alpha_rad)
        A0 = coeff['A0']
        An = coeff['An']
        theta, x_ch, gamma = compute_gamma_distribution(A0, An, V_inf)

        # Normalise by V_inf for non-dimensional plot
        gamma_nd = gamma / V_inf

        self._clear_fig()
        ax1 = self.fig.add_subplot(211)
        ax1.plot(np.rad2deg(theta), gamma_nd, color=COLORS[4], lw=2.5)
        ax1.set_xlabel("θ (deg)")
        ax1.set_ylabel("γ(θ) / V∞")
        ax1.set_title(f"Circulation Distribution γ(θ) at α = {alpha_deg:.1f}°")
        ax1.grid(True)
        ax1.set_xlim(0, 180)

        ax2 = self.fig.add_subplot(212)
        ax2.plot(x_ch, gamma_nd, color=COLORS[2], lw=2.5)
        ax2.set_xlabel("x/c")
        ax2.set_ylabel("γ(x) / V∞")
        ax2.set_title("Circulation Distribution vs Chord Position")
        ax2.set_xlim(0, 1)
        ax2.grid(True)

        self._draw()
        self._set_status(f"γ(θ) plotted at α = {alpha_deg:.1f}°.")

    # -----------------------------------------------------------------------
    # Velocity Field
    # -----------------------------------------------------------------------

    def _plot_vfield(self):
        if not self._require_airfoil():
            return
        V_inf, alpha_rad, alpha_deg = self._get_flow()
        self._set_status("Computing velocity field (this may take 5-15 s) …")
        self.root.update_idletasks()

        coeff = compute_fourier_coefficients(self.camber_func, alpha_rad)
        A0, An = coeff['A0'], coeff['An']

        X, Z, Vx, Vz, Vmag = compute_velocity_field(
            self.camber_func, A0, An, V_inf, alpha_rad
        )

        self._clear_fig()
        ax = self.fig.add_subplot(111)

        # Streamline / quiver
        skip = 2
        q = ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                        Vx[::skip, ::skip], Vz[::skip, ::skip],
                        Vmag[::skip, ::skip],
                        cmap='coolwarm', scale=None,
                        width=0.002, alpha=0.85)
        self.fig.colorbar(q, ax=ax, label='|V| (m/s)', fraction=0.03)

        # Camber line
        x_c, z_c = get_camber_points(self.camber_func, 200)
        ax.plot(x_c, z_c, 'w-', lw=2.5, label='Camber line', zorder=5)

        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("x/c")
        ax.set_ylabel("z/c")
        ax.set_title(f"Velocity Field (4c × 3c)   α = {alpha_deg:.1f}°,  V∞ = {V_inf:.1f} m/s")
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.grid(True)
        self._draw()
        self._set_status(f"Velocity field plotted at α = {alpha_deg:.1f}°.")

    # -----------------------------------------------------------------------
    # Circulation computations
    # -----------------------------------------------------------------------

    def _compute_bound_gamma(self):
        if not self._require_airfoil():
            return
        V_inf, alpha_rad, alpha_deg = self._get_flow()
        self._set_status("Computing bound circulation (distribution integral) …")

        coeff = compute_fourier_coefficients(self.camber_func, alpha_rad)
        A0, An = coeff['A0'], coeff['An']
        Gamma_dist = compute_bound_circulation(A0, An[0], V_inf)
        Cl = compute_Cl(A0, An[0])

        msg = (f"α = {alpha_deg:.2f}°\n"
               f"Cl = {Cl:.5f}\n"
               f"Γ (distribution integral) = {Gamma_dist:.5f} m²/s\n"
               f"  (= V∞·c·Cl/2 = {V_inf*1.0*Cl/2:.5f})")
        self.circulation_text.set(msg)
        self._set_status(f"Bound Γ = {Gamma_dist:.4f} m²/s at α = {alpha_deg:.1f}°")

    def _compute_line_integral(self):
        if not self._require_airfoil():
            return
        V_inf, alpha_rad, alpha_deg = self._get_flow()
        self._set_status("Computing circulation via velocity line integral (may take ~15 s) …")
        self.root.update_idletasks()

        coeff = compute_fourier_coefficients(self.camber_func, alpha_rad)
        A0, An = coeff['A0'], coeff['An']
        Gamma_line = compute_circulation_line_integral(
            self.camber_func, A0, An, V_inf, alpha_rad
        )
        Gamma_dist = compute_bound_circulation(A0, An[0], V_inf)

        prev = self.circulation_text.get()
        msg = (f"{prev}\n"
               f"Γ (velocity line integral)  = {Gamma_line:.5f} m²/s\n"
               f"  Difference from dist. method: {abs(Gamma_line-Gamma_dist)*100/max(abs(Gamma_dist),1e-9):.2f}%")
        self.circulation_text.set(msg)
        self._set_status(f"Line integral Γ = {Gamma_line:.4f} m²/s at α = {alpha_deg:.1f}°")

    # -----------------------------------------------------------------------
    # Custom Design Comparisons
    # -----------------------------------------------------------------------

    def _require_designs(self) -> bool:
        if all(f is None for f in self.design_funcs):
            messagebox.showwarning("No Designs",
                                    "Please load at least one custom design (3 Designs tab).")
            return False
        return True

    def _all_airfoils(self):
        """Return list of (name, func) for NACA + loaded custom designs."""
        all_af = []
        if self.camber_func:
            naca_txt = self.naca_entry.get().strip()
            all_af.append((f"NACA {naca_txt}", self.camber_func))
        for name, func in zip(self.design_names, self.design_funcs):
            if func is not None:
                all_af.append((name, func))
        return all_af

    def _plot_compare_camber(self):
        if not self._require_designs():
            return
        all_af = self._all_airfoils()
        self._clear_fig()
        ax = self.fig.add_subplot(111)
        for idx, (name, func) in enumerate(all_af):
            x, z = get_camber_points(func)
            ax.plot(x, z, color=COLORS[idx % len(COLORS)], lw=2, label=name)
        ax.axhline(0, color='#585b70', lw=0.6, linestyle='--')
        ax.set_xlabel("x/c")
        ax.set_ylabel("z/c")
        ax.set_title("Camber Line Comparison")
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True)
        self._draw()
        self._set_status("Camber comparison plotted.")

    def _plot_compare_Cl(self):
        if not self._require_designs():
            return
        all_af = self._all_airfoils()
        self._clear_fig()
        ax = self.fig.add_subplot(111)
        a_min, a_max = self.alpha_min_v.get(), self.alpha_max_v.get()
        for idx, (name, func) in enumerate(all_af):
            sw = sweep_alpha(func, a_min, a_max, self.n_alpha_v.get())
            ax.plot(sw['alpha_deg'], sw['Cl'],
                     color=COLORS[idx % len(COLORS)], lw=2, label=name)
        ax.axhline(0, color='#585b70', lw=0.6, linestyle='--')
        ax.set_xlabel("α (deg)")
        ax.set_ylabel("Cl")
        ax.set_title("Lift Coefficient Comparison   Cl vs α")
        ax.legend()
        ax.grid(True)
        self._draw()
        self._set_status("Cl comparison plotted.")

    def _plot_compare_Cm(self):
        if not self._require_designs():
            return
        all_af = self._all_airfoils()
        self._clear_fig()
        ax = self.fig.add_subplot(111)
        a_min, a_max = self.alpha_min_v.get(), self.alpha_max_v.get()
        for idx, (name, func) in enumerate(all_af):
            sw = sweep_alpha(func, a_min, a_max, self.n_alpha_v.get())
            ax.plot(sw['alpha_deg'], sw['Cm'],
                     color=COLORS[idx % len(COLORS)], lw=2, label=name)
        ax.axhline(0, color='#585b70', lw=0.6, linestyle='--')
        ax.set_xlabel("α (deg)")
        ax.set_ylabel("Cm (about c/4)")
        ax.set_title("Moment Coefficient Comparison   Cm vs α")
        ax.legend()
        ax.grid(True)
        self._draw()
        self._set_status("Cm comparison plotted.")

    def _plot_compare_vfields(self):
        if not self._require_designs():
            return
        all_af = self._all_airfoils()[:4]   # max 4 (2×2 grid)
        V_inf, alpha_rad, alpha_deg = self._get_flow()

        n = len(all_af)
        ncols = min(n, 2)
        nrows = (n + 1) // 2

        self._clear_fig()
        self._set_status(f"Computing {n} velocity fields … (may take ~30 s)")
        self.root.update_idletasks()

        for idx, (name, func) in enumerate(all_af):
            ax = self.fig.add_subplot(nrows, ncols, idx + 1)
            coeff = compute_fourier_coefficients(func, alpha_rad)
            A0, An = coeff['A0'], coeff['An']
            X, Z, Vx, Vz, Vmag = compute_velocity_field(func, A0, An, V_inf, alpha_rad,
                                                          nx=40, nz=30)
            skip = 2
            ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                       Vx[::skip, ::skip], Vz[::skip, ::skip],
                       Vmag[::skip, ::skip],
                       cmap='coolwarm', scale=None, width=0.003, alpha=0.85)
            xc, zc = get_camber_points(func, 150)
            ax.plot(xc, zc, 'w-', lw=2)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(name, fontsize=9)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)

        self.fig.suptitle(f"Velocity Fields Comparison   α = {alpha_deg:.1f}°", y=1.01)
        self._draw()
        self._set_status("Velocity field comparison plotted.")

    # -----------------------------------------------------------------------
    # Bonus: Pressure Distribution
    # -----------------------------------------------------------------------

    def _plot_pressure(self):
        if not self._require_airfoil():
            return
        V_inf, alpha_rad, alpha_deg = self._get_flow()
        self._set_status("Computing pressure distribution …")

        coeff = compute_fourier_coefficients(self.camber_func, alpha_rad)
        A0, An = coeff['A0'], coeff['An']
        x_arr, Cp_u, Cp_l, Cp_diff, apg_x, apg_v = compute_pressure_distribution(
            A0, An, V_inf, alpha_rad
        )

        self._clear_fig()
        ax1 = self.fig.add_subplot(211)
        ax1.plot(x_arr, Cp_u, color=COLORS[0], lw=2.5, label='Cp_upper (suction)')
        ax1.plot(x_arr, Cp_l, color=COLORS[1], lw=2.5, label='Cp_lower (pressure)')
        ax1.axhline(0, color='#585b70', lw=0.7, linestyle='--')
        if len(apg_x) > 0:
            ax1.fill_betweenx([Cp_u.min(), 0], apg_x[0], apg_x[-1],
                               alpha=0.15, color='#f38ba8', label='APG region (upper)')
        ax1.invert_yaxis()   # convention: suction (−Cp) plotted upward
        ax1.set_xlabel("x/c")
        ax1.set_ylabel("Cp")
        ax1.set_title(f"Pressure Distribution at α = {alpha_deg:.1f}°")
        ax1.legend(fontsize=8)
        ax1.grid(True)
        ax1.set_xlim(0, 1)

        ax2 = self.fig.add_subplot(212)
        ax2.plot(x_arr, Cp_diff, color=COLORS[4], lw=2.5, label='ΔCp = Cp_lower − Cp_upper')
        ax2.axhline(0, color='#585b70', lw=0.7, linestyle='--')
        ax2.set_xlabel("x/c")
        ax2.set_ylabel("ΔCp")
        ax2.set_title("Pressure Difference  ΔCp(x)  (proportional to γ/V∞)")
        ax2.legend(fontsize=8)
        ax2.grid(True)
        ax2.set_xlim(0, 1)

        self._draw()
        self._set_status(f"Pressure distribution plotted at α = {alpha_deg:.1f}°.")

    def _rank_apg(self):
        from pressure import find_minimum_apg_camber
        all_af = self._all_airfoils()
        if not all_af:
            messagebox.showwarning("No Airfoils", "Load at least one airfoil first.")
            return
        _, alpha_rad, alpha_deg = self._get_flow()
        results = find_minimum_apg_camber(all_af, alpha_rad)
        lines = [f"APG Ranking at α = {alpha_deg:.1f}°", "=" * 30]
        for rank, (name, rms_apg, Cl) in enumerate(results, 1):
            lines.append(f"{rank}. {name:15s}  RMS-APG={rms_apg:.4f}  Cl={Cl:.4f}")
        self.apg_text.set('\n'.join(lines))
        self._set_status("APG ranking complete.")


# ============================================================================
# Entry point
# ============================================================================

def main():
    root = tk.Tk()

    # Apply ttk theme
    style = ttk.Style(root)
    available = style.theme_names()
    for theme in ('clam', 'alt', 'default'):
        if theme in available:
            style.theme_use(theme)
            break

    style.configure('TNotebook', background=PANEL_BG, borderwidth=0)
    style.configure('TNotebook.Tab', background='#313244', foreground=FG,
                     padding=[8, 4], font=('Segoe UI', 9))
    style.map('TNotebook.Tab',
               background=[('selected', '#45475a')],
               foreground=[('selected', '#cba6f7')])
    style.configure('TFrame', background=PANEL_BG)
    style.configure('TScale', background=PANEL_BG, troughcolor='#45475a')

    app = AirfoilApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

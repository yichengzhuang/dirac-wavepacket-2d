#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical unit conversion layer for the 2D Dirac simulation.

Usage (optional — the simulation runs fine without this module):

    from units import PhysicalUnits

    # Define a physical system
    pu = PhysicalUnits(
        vF_ms=1e6,           # Fermi velocity in m/s
        dot_size_nm=50,      # quantum dot semi-axis 'a' in nm
        mass_gap_meV=30,     # mass gap 2m in meV (0 for massless)
    )

    # Get simulation parameters in natural units
    print(pu.summary())
    a_code = pu.length_to_code(50)    # 50 nm → code units
    m_code = pu.mass_gap_to_code(30)  # 30 meV gap → code mass parameter

    # After simulation, convert back
    E_meV = pu.energy_to_physical(2.4)  # code energy → meV
    t_fs  = pu.time_to_physical(100)    # code time → femtoseconds
    Gamma_meV = pu.energy_to_physical(0.1)

All simulation code remains in natural units (ℏ = 1, vF = 1).
This module only provides the conversion factors.
"""

import numpy as np


# Physical constants
HBAR_EV_S = 6.582119569e-16   # ℏ in eV·s
HBAR_J_S  = 1.054571817e-34   # ℏ in J·s
NM_PER_M  = 1e9
FS_PER_S  = 1e15


class PhysicalUnits:
    """Bidirectional conversion between physical units and simulation natural units.

    Natural units in the simulation: ℏ = 1, vF = 1.
    One code length unit = L0 nm (set by mapping dot_size_nm to the code parameter 'a').

    Parameters
    ----------
    vF_ms : float
        Fermi velocity in m/s.  Graphene: ~1e6, TI surface: ~5e5.
    dot_size_nm : float
        Physical size of the quantum dot semi-axis in nm.
        This is mapped to the code parameter 'a' (default a=2).
    mass_gap_meV : float
        Mass gap Δ = 2m·vF² in meV.  0 for massless.
    code_a : float
        The semi-axis in code units (default 2.0, matching main.py).
    """

    def __init__(self, vF_ms=1e6, dot_size_nm=50.0, mass_gap_meV=0.0, code_a=2.0):
        self.vF_ms = vF_ms
        self.dot_size_nm = dot_size_nm
        self.mass_gap_meV = mass_gap_meV
        self.code_a = code_a

        # 1 code length unit = L0 nm
        self.L0_nm = dot_size_nm / code_a
        self.L0_m  = self.L0_nm / NM_PER_M

        # Energy unit: ℏ vF / L0
        self.hbar_vF_eVnm = HBAR_EV_S * vF_ms * NM_PER_M  # ℏvF in eV·nm
        self.E0_eV  = self.hbar_vF_eVnm / self.L0_nm       # energy unit in eV
        self.E0_meV = self.E0_eV * 1e3                      # energy unit in meV

        # Time unit: ℏ / E0 = L0 / vF
        self.T0_s  = self.L0_m / vF_ms
        self.T0_fs = self.T0_s * FS_PER_S

        # Mass parameter in code units: gap = 2m in energy units
        # m_code = gap_meV / (2 * E0_meV)
        if mass_gap_meV > 0:
            self.m_code = mass_gap_meV / (2.0 * self.E0_meV)
        else:
            self.m_code = 0.0

    # ── Code → Physical ──

    def energy_to_physical(self, E_code):
        """Convert energy from code units to meV."""
        return np.asarray(E_code) * self.E0_meV

    def time_to_physical(self, t_code):
        """Convert time from code units to femtoseconds."""
        return np.asarray(t_code) * self.T0_fs

    def length_to_physical(self, x_code):
        """Convert length from code units to nm."""
        return np.asarray(x_code) * self.L0_nm

    def width_to_physical(self, Gamma_code):
        """Convert resonance width Γ from code units to meV."""
        return self.energy_to_physical(Gamma_code)

    # ── Physical → Code ──

    def energy_to_code(self, E_meV):
        """Convert energy from meV to code units."""
        return np.asarray(E_meV) / self.E0_meV

    def time_to_code(self, t_fs):
        """Convert time from femtoseconds to code units."""
        return np.asarray(t_fs) / self.T0_fs

    def length_to_code(self, x_nm):
        """Convert length from nm to code units."""
        return np.asarray(x_nm) / self.L0_nm

    def mass_gap_to_code(self, gap_meV):
        """Convert mass gap (2m·vF²) in meV to code mass parameter m."""
        return gap_meV / (2.0 * self.E0_meV)

    # ── Summary ──

    def summary(self):
        """Return a formatted summary of unit conversions."""
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║         Physical Unit Conversion             ║",
            "╚══════════════════════════════════════════════╝",
            f"  vF           = {self.vF_ms:.2e} m/s",
            f"  dot size     = {self.dot_size_nm:.1f} nm  (code a = {self.code_a})",
            f"  ℏvF          = {self.hbar_vF_eVnm:.4f} eV·nm",
            f"",
            f"  1 code length = {self.L0_nm:.2f} nm",
            f"  1 code energy = {self.E0_meV:.2f} meV",
            f"  1 code time   = {self.T0_fs:.2f} fs",
            f"",
        ]
        if self.mass_gap_meV > 0:
            lines.append(f"  Mass gap     = {self.mass_gap_meV:.1f} meV")
            lines.append(f"  → code m     = {self.m_code:.4f}")
        else:
            lines.append(f"  Mass gap     = 0 (massless)")
            lines.append(f"  → code m     = 0")
        return "\n".join(lines)

    def results_table(self, E_code, Gamma_code, labels=None):
        """Format extracted resonances in physical units.

        Parameters
        ----------
        E_code, Gamma_code : array-like
        labels : list of str, optional

        Returns
        -------
        str : formatted table
        """
        E_code = np.atleast_1d(E_code)
        Gamma_code = np.atleast_1d(Gamma_code)
        n = len(E_code)
        lines = [f"  {'Mode':>6} | {'E (code)':>10} {'E (meV)':>10} | "
                 f"{'Γ (code)':>10} {'Γ (meV)':>10} {'Q':>8}"]
        lines.append("  " + "-" * 68)
        for j in range(n):
            E_meV = self.energy_to_physical(E_code[j])
            G_meV = self.energy_to_physical(Gamma_code[j])
            Q = abs(E_code[j]) / Gamma_code[j] if Gamma_code[j] > 0 else float('inf')
            lbl = labels[j] if labels else f"{j+1}"
            lines.append(f"  {lbl:>6} | {E_code[j]:10.4f} {E_meV:10.2f} | "
                         f"{Gamma_code[j]:10.4f} {G_meV:10.2f} {Q:8.1f}")
        return "\n".join(lines)


# ── Preset systems ──

def graphene_small_dot(dot_nm=20, gap_meV=0):
    """Graphene quantum dot, ~20 nm, electrostatically gated."""
    return PhysicalUnits(vF_ms=1e6, dot_size_nm=dot_nm, mass_gap_meV=gap_meV)

def graphene_hBN(dot_nm=50, gap_meV=40):
    """Graphene on hBN substrate (sublattice gap ~30-50 meV)."""
    return PhysicalUnits(vF_ms=1e6, dot_size_nm=dot_nm, mass_gap_meV=gap_meV)

def TI_surface(dot_nm=30, gap_meV=20):
    """Topological insulator surface state (Bi₂Se₃), magnetic gap."""
    return PhysicalUnits(vF_ms=5e5, dot_size_nm=dot_nm, mass_gap_meV=gap_meV)

def TMD_dot(dot_nm=5, gap_meV=1600):
    """Transition metal dichalcogenide quantum dot (MoS₂)."""
    return PhysicalUnits(vF_ms=5e5, dot_size_nm=dot_nm, mass_gap_meV=gap_meV)


if __name__ == "__main__":
    print("=== Example: Graphene on hBN, 50 nm dot ===\n")
    pu = graphene_hBN(dot_nm=50, gap_meV=40)
    print(pu.summary())
    print()

    # Simulated resonance results (from m=0.5 run)
    E_code = [2.01, 2.54, 3.07]
    G_code = [0.300, 0.224, 0.219]
    print(pu.results_table(E_code, G_code))

    print("\n=== Preset systems with m=1 in code units ===\n")
    for name, fn in [("Graphene 20nm", lambda: graphene_small_dot(20)),
                     ("Graphene 50nm", lambda: graphene_small_dot(50)),
                     ("Graphene/hBN 50nm", lambda: graphene_hBN(50, 40)),
                     ("TI surface 30nm", lambda: TI_surface(30, 20)),
                     ("TMD 5nm", lambda: TMD_dot(5, 1600))]:
        pu = fn()
        gap_for_m1 = 2 * pu.E0_meV  # gap when m_code = 1
        print(f"  {name:25s}: 1 code E = {pu.E0_meV:8.2f} meV, "
              f"m=1 → gap = {gap_for_m1:.0f} meV")

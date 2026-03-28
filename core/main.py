#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main driver for 2D Dirac + CAP + harmonic-inversion resonance extraction.
"""

import os
import numpy as np
from datetime import datetime

from dynamics import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, density, overlap_R,
    propagate,
)
from spectral import (
    extract_resonances, filter_modes, symmetrise_modes,
    initial_energy_envelope,
)
from visualization import (
    plot_initial_wavepacket, plot_potential, plot_CAP, plot_mask,
    plot_correlation, plot_energies, plot_widths, plot_spectrum_overlay,
    plot_filtered_modes, plot_symmetry_modes, plot_late_time,
    plot_transmission,
    playback_animation, AnimationRecorder, LiveDisplay,
)


# ╔═══════════════════════════════════════════════════════════════╗
# ║                   PIPELINE SWITCHES                          ║
# ╚═══════════════════════════════════════════════════════════════╝

RUN_CORRELATION      = True
RUN_SPECTRUM         = True   # needs RUN_CORRELATION
RUN_MODE_FILTER      = True   # needs RUN_SPECTRUM
MEASURE_TRANSMISSION = True

SHOW_STATIC_PLOTS = True
PRINT_MODE_TABLE  = True

# ╔═══════════════════════════════════════════════════════════════╗
# ║                   ANIMATION SETTINGS                         ║
# ╚═══════════════════════════════════════════════════════════════╝
# LIVE:     update plot during propagation  (plt.ion)
# PLAYBACK: smooth replay after propagation (FuncAnimation)
# SAVE:     stream to mp4/gif during propagation (file only)

LIVE_DISPLAY        = True
LIVE_EVERY          = 20

PLAYBACK_ANIMATION  = False
PLAYBACK_STRIDE     = 1
PLAYBACK_INTERVAL   = 30

SAVE_ANIMATION      = False
SAVE_ANIM_FILE      = "propagation.gif"
SAVE_ANIM_FPS       = 30
SAVE_ANIM_EVERY     = 100

# ╔═══════════════════════════════════════════════════════════════╗
# ║                      RUN MODE                                ║
# ╚═══════════════════════════════════════════════════════════════╝

MODE = "single"          # "single" or "symmetry"


# ╔═══════════════════════════════════════════════════════════════╗
# ║                   PHYSICS PARAMETERS                         ║
# ╚═══════════════════════════════════════════════════════════════╝

# ── grid ──
n  = 4*128
L  = 10.0

# ── Dirac ──
vF = 1.0
m  = 0.0

# ── time ──
dt      = 0.01
n_steps = 10000

# ── recording cadence ──
decim      = 10        # record C_R every decim steps (dt_corr = dt*decim)
save_every = 10        # snapshot cadence (must be multiple of decim)
max_record = 2500      # cap recorded samples (memory safety)

# ── potential ──
POTENTIAL_OPTION = "harmonic_ellipse"

if POTENTIAL_OPTION == "hard_circle":
    pot_params = dict(radius=2.0, V0=5e3, smooth_width=0.03)
elif POTENTIAL_OPTION == "hard_ellipse":
    pot_params = dict(a=2.0, b=1.2, V0=5e3, smooth_width=0.03)
elif POTENTIAL_OPTION == "soft_circle":
    pot_params = dict(radius=20.0, V0=70.0, lam=25.0, mu=5.0)
elif POTENTIAL_OPTION == "soft_ellipse":
    pot_params = dict(a=20.0, b=12.0, V0=70.0, lam=25.0, mu=5.0)
elif POTENTIAL_OPTION == "soft_stadium":
    pot_params = dict(a_half=20.0, b=10.0, V0=70.0, lam=25.0, mu=5.0)
elif POTENTIAL_OPTION == "harmonic_circle":
    pot_params = dict(k_harm=3.0, r0=20.0)
elif POTENTIAL_OPTION == "harmonic_ellipse":
    pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
else:
    raise ValueError(f"Unknown POTENTIAL_OPTION: {POTENTIAL_OPTION}")

# ── CAP ──
cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

# ── analysis mask ──
mask_safety_factor = 0.95
mask_margin        = 1.30

# ── initial wavepacket ──
x0, y0     = 0.0, 0.0
kx0, ky0   = -1.0, 0.0
sigma0     = 0.6
theta0     = None          # None → chiral; float → fixed phase
band0      = "conduction"

# ── spectral extraction ──
Gamma_cut       = 8.0
pencil_fraction = 4
L_pencil_max    = 400
n_modes         = 3

# ── transmission ──
transmission_every = 10
auto_stop_T        = None  # stop when T >= this value (e.g. 0.99); None = run all steps


# ╔═══════════════════════════════════════════════════════════════╗
# ║                     OUTPUT FOLDER                            ║
# ╚═══════════════════════════════════════════════════════════════╝

def _spinor_tag():
    if theta0 is None:
        return f"chiral_{band0}"
    return f"fixed_theta{np.degrees(theta0):.1f}deg"

def make_output_folder():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _here = os.path.dirname(os.path.abspath(__file__))
    name = os.path.join(_here, "Results",
        f"{ts}__{POTENTIAL_OPTION}__"
        f"{_spinor_tag()}__"
        f"kx={kx0}_ky={ky0}__"
        f"n={n}_dt={dt}_T={n_steps*dt:.0f}"
    )
    os.makedirs(name, exist_ok=True)
    return name


# ╔═══════════════════════════════════════════════════════════════╗
# ║               SINGLE-k RUN FUNCTION                          ║
# ╚═══════════════════════════════════════════════════════════════╝

def run_single(kx, ky, label, folder, grids, V, W, chi_R, mask_info):
    X, Y, KX, KY, K, dx = grids
    sqrt_chi = np.sqrt(chi_R)

    halfV, kinetic = build_split_operator(vF, dt, m, V, W, KX, KY, K)

    psi0 = gaussian_wavepacket(
        X, Y, x0=x0, y0=y0, kx0=kx, ky0=ky,
        sigma=sigma0, theta=theta0, band=band0, dx=dx,
    )

    tag = f"_{label}"
    sub = os.path.join(folder, label)
    os.makedirs(sub, exist_ok=True)

    if SHOW_STATIC_PLOTS:
        plot_initial_wavepacket(psi0, L, folder=sub, tag=tag)
        plot_potential(V, L, title=f"V: {POTENTIAL_OPTION}", folder=sub, tag=tag)
        plot_CAP(W, L, folder=sub, tag=tag)
        plot_mask(chi_R, L, mask_info, folder=sub, tag=tag)

    # ── streaming file animation ──
    rec = None; anim_writer = None; anim_cb = None
    if SAVE_ANIMATION:
        anim_path = os.path.join(sub, SAVE_ANIM_FILE)
        rec = AnimationRecorder(anim_path, L,
                                chi_R=chi_R, cap_radius=cap_radius,
                                fps=SAVE_ANIM_FPS)
        anim_writer = rec.writer
        anim_cb = rec.update

    # ── live display ──
    disp = None; live_cb = None
    if LIVE_DISPLAY:
        disp = LiveDisplay(L, chi_R=chi_R, cap_radius=cap_radius,
                           title_fmt=f"[{label}] t = {{t:.3f}}")
        live_cb = disp.update

    # ── propagate ──
    print(f"\n  [{label}] Propagating {n_steps} steps "
          f"(decim={decim}, dt_corr={dt*decim:.4g}) …")
    result = propagate(
        psi0, halfV, kinetic, sqrt_chi, dx,
        n_steps=n_steps, dt=dt,
        save_every=save_every,
        corr_every=decim,
        max_record=max_record,
        live_callback=live_cb,
        live_every=LIVE_EVERY,
        anim_writer=anim_writer,
        anim_callback=anim_cb,
        anim_every=SAVE_ANIM_EVERY,
        measure_transmission=MEASURE_TRANSMISSION,
        transmission_every=transmission_every,
        auto_stop_T=auto_stop_T,
    )

    if disp is not None:
        disp.close()
    if rec is not None:
        rec.finish()

    C_R        = result["C_R"]
    corr_times = result["corr_times"]
    dt_corr    = result["dt_corr"]
    snapshots  = result["snapshots"]
    snap_times = result["snap_times"]

    print(f"  [{label}] Recorded {len(C_R)} correlation samples, "
          f"{len(snapshots)} snapshots")

    # ── playback animation ──
    if PLAYBACK_ANIMATION:
        print(f"  [{label}] Playing animation …")
        save_pb = os.path.join(sub, f"playback{tag}.mp4") if SAVE_ANIMATION else None
        playback_animation(
            snapshots, snap_times, L,
            chi_R=chi_R, cap_radius=cap_radius,
            stride=PLAYBACK_STRIDE,
            interval_ms=PLAYBACK_INTERVAL,
            save_path=save_pb, save_fps=SAVE_ANIM_FPS,
        )

    # ── correlation plot ──
    if RUN_CORRELATION and SHOW_STATIC_PLOTS:
        plot_correlation(corr_times, C_R, folder=sub, tag=tag)

    # ── transmission ──
    if MEASURE_TRANSMISSION and "P_inside" in result:
        if SHOW_STATIC_PLOTS:
            plot_transmission(result["P_times"], result["P_inside"],
                              folder=sub, tag=tag)
        P0 = result["P_inside"][0]
        Pf = result["P_inside"][-1]
        if P0 > 0:
            print(f"  [{label}] Transmission:  "
                  f"P(0)={P0:.4f}  P(T)={Pf:.4f}  T={1-Pf/P0:.4f}")

    # ── harmonic inversion ──
    E = Gamma = omega = a = np.array([])
    if RUN_SPECTRUM:
        if not RUN_CORRELATION:
            print("  WARNING: RUN_SPECTRUM needs RUN_CORRELATION; skipping.")
        else:
            print(f"  [{label}] Harmonic inversion (dt_corr={dt_corr:.4g}) …")
            spec = extract_resonances(
                C_R, corr_times, dt_corr,
                Gamma_cut=Gamma_cut,
                pencil_fraction=pencil_fraction,
                L_pencil_max=L_pencil_max,
            )
            E     = spec["E"]
            Gamma = spec["Gamma"]
            omega = spec["omega"]
            a     = spec["a"]

            if SHOW_STATIC_PLOTS:
                plot_energies(E, folder=sub, tag=tag)
                plot_widths(Gamma, folder=sub, tag=tag)
                Egrid, env = initial_energy_envelope(psi0, KX, KY, vF=vF, m=m)
                plot_spectrum_overlay(E, a, Egrid, env, folder=sub, tag=tag)

    # ── mode filtering ──
    modes = []
    if RUN_MODE_FILTER and len(omega) > 0:
        nm = min(n_modes, len(omega))
        print(f"  [{label}] Filtering {nm} modes …")
        modes = filter_modes(snapshots, snap_times, omega, chi_R, n_modes=nm)
        if SHOW_STATIC_PLOTS:
            plot_filtered_modes(modes, L, E=E, folder=sub, tag=tag)
            plot_late_time(snapshots[-1], chi_R, L, folder=sub, tag=tag)

    if PRINT_MODE_TABLE and len(E) > 0:
        for j in range(min(n_modes, len(E))):
            print(f"    mode {j+1}:  E={E[j]:.6f}  "
                  f"Γ={Gamma[j]:.6f}  |a|={abs(a[j]):.4e}")

    return {
        "psi0": psi0, "E": E, "Gamma": Gamma,
        "omega": omega, "a": a, "modes": modes,
        "snapshots": snapshots, "snap_times": snap_times,
        "P_inside": result.get("P_inside"),
        "P_times":  result.get("P_times"),
    }


# ╔═══════════════════════════════════════════════════════════════╗
# ║                        MAIN                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

def main():
    folder = make_output_folder()
    print(f"Output → {folder}/")

    grids = make_grid(n, L)
    X, Y, KX, KY, K, dx = grids
    V = build_potential(POTENTIAL_OPTION, X, Y, pot_params)
    W = radial_CAP(X, Y, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power)
    chi_R, mask_info = build_analysis_mask(
        POTENTIAL_OPTION, X, Y, pot_params,
        cap_radius=cap_radius,
        safety_factor=mask_safety_factor,
        mask_margin=mask_margin,
    )

    if MODE == "single":
        run_single(kx0, ky0, label="single", folder=folder,
                   grids=grids, V=V, W=W, chi_R=chi_R, mask_info=mask_info)

    elif MODE == "symmetry":
        res_plus = run_single(
            +kx0, +ky0, label="plus_k", folder=folder,
            grids=grids, V=V, W=W, chi_R=chi_R, mask_info=mask_info)
        res_minus = run_single(
            -kx0, -ky0, label="minus_k", folder=folder,
            grids=grids, V=V, W=W, chi_R=chi_R, mask_info=mask_info)

        if RUN_MODE_FILTER and len(res_plus["modes"]) > 0:
            print("\n  Building symmetry-adapted (even/odd) modes …")
            sym = symmetrise_modes(res_plus["modes"], res_minus["modes"])
            plot_symmetry_modes(sym, L, folder=folder, tag="_sym")
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    print(f"\nDone.  All outputs in:  {folder}/")


if __name__ == "__main__":
    main()
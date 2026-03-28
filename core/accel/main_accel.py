#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accelerated main driver for 2D Dirac + CAP + harmonic-inversion.

Mirrors main.py but uses the backend-abstracted dynamics for CPU/GPU
acceleration.  spectral.py and visualization.py are reused unchanged.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Ensure parent directory is on path for spectral/visualization imports
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
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
# ║                   BACKEND SELECTION                           ║
# ╚═══════════════════════════════════════════════════════════════╝

BACKEND = "auto"   # "auto", "torch_mps", "numpy_scipy", "numpy"


# ╔═══════════════════════════════════════════════════════════════╗
# ║                   PIPELINE SWITCHES                          ║
# ╚═══════════════════════════════════════════════════════════════╝

RUN_CORRELATION      = True
RUN_SPECTRUM         = True
RUN_MODE_FILTER      = True
MEASURE_TRANSMISSION = True

SHOW_STATIC_PLOTS = True
PRINT_MODE_TABLE  = True

# ╔═══════════════════════════════════════════════════════════════╗
# ║                   ANIMATION SETTINGS                         ║
# ╚═══════════════════════════════════════════════════════════════╝

LIVE_DISPLAY        = False
LIVE_EVERY          = 20

PLAYBACK_ANIMATION  = False
PLAYBACK_STRIDE     = 1
PLAYBACK_INTERVAL   = 30

SAVE_ANIMATION      = True
SAVE_ANIM_FILE      = "propagation.gif"
SAVE_ANIM_FPS       = 30
SAVE_ANIM_EVERY     = 100

# ╔═══════════════════════════════════════════════════════════════╗
# ║                      RUN MODE                                ║
# ╚═══════════════════════════════════════════════════════════════╝

MODE = "single"

# ╔═══════════════════════════════════════════════════════════════╗
# ║                   PHYSICS PARAMETERS                         ║
# ╚═══════════════════════════════════════════════════════════════╝

# ── grid ──
n  = 512
L  = 10.0

# ── Dirac ──
vF = 1.0
m  = 0.0

# ── time ──
dt      = 0.01
n_steps = 10000

# ── recording cadence ──
decim      = 10
save_every = 10
max_record = 2500

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
theta0     = None
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
    name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Results",
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

def run_single(kx, ky, label, folder, X, Y, KX, KY, K, dx,
               X_np, Y_np, V, W, chi_R, mask_info, xp):

    sqrt_chi = xp.sqrt(chi_R)

    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)

    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=x0, y0=y0, kx0=kx, ky0=ky,
        sigma=sigma0, theta=theta0, band=band0, dx=dx, xp=xp,
    )

    tag = f"_{label}"
    sub = os.path.join(folder, label)
    os.makedirs(sub, exist_ok=True)

    # Static plots need CPU arrays
    if SHOW_STATIC_PLOTS:
        psi0_cpu = np.stack([xp.to_numpy(psi0_up), xp.to_numpy(psi0_dn)], axis=0)
        V_cpu = xp.to_numpy(V)
        W_cpu = xp.to_numpy(W)
        chi_cpu = xp.to_numpy(chi_R)
        plot_initial_wavepacket(psi0_cpu, L, folder=sub, tag=tag)
        plot_potential(V_cpu, L, title=f"V: {POTENTIAL_OPTION}", folder=sub, tag=tag)
        plot_CAP(W_cpu, L, folder=sub, tag=tag)
        plot_mask(chi_cpu, L, mask_info, folder=sub, tag=tag)

    # ── streaming file animation ──
    rec = None; anim_writer = None; anim_cb = None
    if SAVE_ANIMATION:
        anim_path = os.path.join(sub, SAVE_ANIM_FILE)
        rec = AnimationRecorder(anim_path, L,
                                chi_R=xp.to_numpy(chi_R), cap_radius=cap_radius,
                                fps=SAVE_ANIM_FPS)
        anim_writer = rec.writer
        anim_cb = rec.update

    # ── live display ──
    disp = None; live_cb = None
    if LIVE_DISPLAY:
        disp = LiveDisplay(L, chi_R=xp.to_numpy(chi_R), cap_radius=cap_radius,
                           title_fmt=f"[{label}] t = {{t:.3f}}")
        live_cb = disp.update

    # ── propagate ──
    print(f"\n  [{label}] Propagating {n_steps} steps "
          f"(decim={decim}, dt_corr={dt*decim:.4g}, backend={xp.name}) …")
    result = propagate(
        psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
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
            chi_R=xp.to_numpy(chi_R), cap_radius=cap_radius,
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
                # initial_energy_envelope needs CPU psi0
                psi0_cpu = np.stack([xp.to_numpy(psi0_up), xp.to_numpy(psi0_dn)], axis=0)
                KX_np = xp.to_numpy(KX)
                KY_np = xp.to_numpy(KY)
                Egrid, env = initial_energy_envelope(psi0_cpu, KX_np, KY_np, vF=vF, m=m)
                plot_spectrum_overlay(E, a, Egrid, env, folder=sub, tag=tag)

    # ── mode filtering ──
    modes = []
    chi_cpu = xp.to_numpy(chi_R)
    if RUN_MODE_FILTER and len(omega) > 0:
        nm = min(n_modes, len(omega))
        print(f"  [{label}] Filtering {nm} modes …")
        modes = filter_modes(snapshots, snap_times, omega, chi_cpu, n_modes=nm)
        if SHOW_STATIC_PLOTS:
            plot_filtered_modes(modes, L, E=E, folder=sub, tag=tag)
            plot_late_time(snapshots[-1], chi_cpu, L, folder=sub, tag=tag)

    if PRINT_MODE_TABLE and len(E) > 0:
        for j in range(min(n_modes, len(E))):
            print(f"    mode {j+1}:  E={E[j]:.6f}  "
                  f"Γ={Gamma[j]:.6f}  |a|={abs(a[j]):.4e}")

    return {
        "psi0_up": xp.to_numpy(psi0_up),
        "psi0_dn": xp.to_numpy(psi0_dn),
        "E": E, "Gamma": Gamma,
        "omega": omega, "a": a, "modes": modes,
        "snapshots": snapshots, "snap_times": snap_times,
        "P_inside": result.get("P_inside"),
        "P_times":  result.get("P_times"),
    }


# ╔═══════════════════════════════════════════════════════════════╗
# ║                        MAIN                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

def main():
    t_total_start = time.perf_counter()

    xp = get_backend(BACKEND)
    print(f"Backend: {xp.name}")

    folder = make_output_folder()
    print(f"Output → {folder}/")

    # ── grids ──
    # CPU grids for potential/mask construction
    x_np = np.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = float(x_np[1] - x_np[0])
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")

    # Backend grids for propagation
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    # ── potential, CAP, mask (CPU → device) ──
    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, mask_info = build_analysis_mask(
        POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius,
        safety_factor=mask_safety_factor,
        mask_margin=mask_margin,
        xp=xp,
    )

    if MODE == "single":
        run_single(kx0, ky0, label="single", folder=folder,
                   X=X, Y=Y, KX=KX, KY=KY, K=K, dx=dx,
                   X_np=X_np, Y_np=Y_np,
                   V=V, W=W, chi_R=chi_R, mask_info=mask_info, xp=xp)

    elif MODE == "symmetry":
        res_plus = run_single(
            +kx0, +ky0, label="plus_k", folder=folder,
            X=X, Y=Y, KX=KX, KY=KY, K=K, dx=dx,
            X_np=X_np, Y_np=Y_np,
            V=V, W=W, chi_R=chi_R, mask_info=mask_info, xp=xp)
        res_minus = run_single(
            -kx0, -ky0, label="minus_k", folder=folder,
            X=X, Y=Y, KX=KX, KY=KY, K=K, dx=dx,
            X_np=X_np, Y_np=Y_np,
            V=V, W=W, chi_R=chi_R, mask_info=mask_info, xp=xp)

        if RUN_MODE_FILTER and len(res_plus["modes"]) > 0:
            print("\n  Building symmetry-adapted (even/odd) modes …")
            sym = symmetrise_modes(res_plus["modes"], res_minus["modes"])
            plot_symmetry_modes(sym, L, folder=folder, tag="_sym")
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    elapsed = time.perf_counter() - t_total_start
    print(f"\nDone.  Total wall time: {elapsed:.1f} s")
    print(f"All outputs in:  {folder}/")


if __name__ == "__main__":
    main()

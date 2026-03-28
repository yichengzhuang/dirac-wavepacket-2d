#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module: static diagnostic plots, smooth post-propagation
animation playback (FuncAnimation), memory-efficient video/GIF export,
and transmission plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from dynamics import density


# ═══════════════════════════════════════════════
# Save helper
# ═══════════════════════════════════════════════

def _ensure_dir(folder):
    if folder:
        os.makedirs(folder, exist_ok=True)


def _save(fig, folder, name, dpi=200):
    if folder is None:
        plt.show()
        return
    _ensure_dir(folder)
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════
# Static diagnostic plots
# ═══════════════════════════════════════════════

def plot_initial_wavepacket(psi0, L, folder=None, tag=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ext = [-L / 2, L / 2, -L / 2, L / 2]

    im0 = axes[0].imshow(density(psi0), extent=ext, origin="lower")
    fig.colorbar(im0, ax=axes[0]); axes[0].set_title("Initial |ψ|²")

    im1 = axes[1].imshow(np.abs(psi0[0])**2, extent=ext, origin="lower")
    fig.colorbar(im1, ax=axes[1]); axes[1].set_title("|ψ_A|²")

    im2 = axes[2].imshow(np.abs(psi0[1])**2, extent=ext, origin="lower")
    fig.colorbar(im2, ax=axes[2]); axes[2].set_title("|ψ_B|²")

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    _save(fig, folder, f"initial_wavepacket{tag}.png")


def _plot_field(data, L, title, folder, name, figsize=(5.5, 4.6)):
    """Generic 2D imshow plot with colorbar."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, extent=[-L/2, L/2, -L/2, L/2], origin="lower")
    fig.colorbar(im, ax=ax); ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.tight_layout()
    _save(fig, folder, name)


def plot_potential(V, L, title="Potential V(x,y)", folder=None, tag=""):
    _plot_field(V, L, title, folder, f"potential{tag}.png")


def plot_CAP(W, L, folder=None, tag=""):
    _plot_field(W, L, "CAP W(x,y)", folder, f"CAP{tag}.png")


def plot_mask(chi_R, L, mask_info, folder=None, tag=""):
    _plot_field(chi_R, L, f"Analysis mask χ_R ({mask_info})", folder,
                f"analysis_mask{tag}.png", figsize=(6, 4.8))


def plot_correlation(times, C_R, folder=None, tag=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, np.real(C_R), label="Re C_R")
    ax.plot(times, np.imag(C_R), label="Im C_R", alpha=0.6)
    ax.set_title("Regional autocorrelation C_R(t)")
    ax.set_xlabel("t"); ax.legend(); fig.tight_layout()
    _save(fig, folder, f"CR_real_imag{tag}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, np.abs(C_R))
    ax.set_title("|C_R(t)|"); ax.set_xlabel("t"); fig.tight_layout()
    _save(fig, folder, f"CR_abs{tag}.png")


def plot_energies(E, n_show=10, folder=None, tag=""):
    n = min(n_show, len(E))
    if n == 0: return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stem(range(n), E[:n])
    ax.axhline(0, color="r", lw=0.8, alpha=0.5)
    ax.set_title("Extracted energies E_n")
    ax.set_xlabel("mode index"); ax.set_ylabel("E"); fig.tight_layout()
    _save(fig, folder, f"extracted_energies{tag}.png")


def plot_widths(Gamma, n_show=10, folder=None, tag=""):
    n = min(n_show, len(Gamma))
    if n == 0: return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stem(range(n), Gamma[:n])
    ax.axhline(0, color="r", lw=0.8, alpha=0.5)
    ax.set_title("Extracted widths Γ_n")
    ax.set_xlabel("mode index"); ax.set_ylabel("Γ"); fig.tight_layout()
    _save(fig, folder, f"extracted_widths{tag}.png")


def plot_spectrum_overlay(E, a, Egrid, envelope, n_show=10, folder=None, tag=""):
    n = min(n_show, len(E))
    if n == 0: return
    stick_h = np.abs(a[:n]) / (np.max(np.abs(a[:n])) + 1e-15)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(Egrid, envelope, lw=2, label="Initial wavepacket envelope")
    for j in range(n):
        ax.vlines(E[j], 0.0, 0.9 * stick_h[j], lw=2)
    ax.set_title("Spectrum: extracted energies vs initial envelope")
    ax.set_xlabel("Energy E"); ax.set_ylabel("normalised weight")
    ax.set_ylim(0.0, 1.05); ax.legend(); fig.tight_layout()
    _save(fig, folder, f"spectrum_overlay{tag}.png")


def plot_filtered_modes(modes, L, E=None, folder=None, tag=""):
    for j, phi in enumerate(modes):
        elabel = f"  E={E[j]:.4f}" if E is not None and j < len(E) else ""
        _plot_field(density(phi), L, f"Filtered mode {j+1}{elabel}",
                    folder, f"filtered_mode_{j+1}{tag}.png")


def plot_symmetry_modes(sym_pairs, L, folder=None, tag=""):
    for j, (phi_even, phi_odd) in enumerate(sym_pairs):
        for phi, sym in [(phi_even, "EVEN"), (phi_odd, "ODD")]:
            _plot_field(density(phi), L, f"{sym} mode {j+1}",
                        folder, f"sym_{sym.lower()}_mode_{j+1}{tag}.png")


def plot_late_time(snapshots_last, chi_R, L, folder=None, tag=""):
    _plot_field(density(snapshots_last) * chi_R, L,
                "Late-time density (masked)", folder,
                f"late_time_density{tag}.png")


def plot_transmission(P_times, P_inside, folder=None, tag=""):
    P0 = P_inside[0] if P_inside[0] > 0 else 1.0
    T_frac = 1.0 - P_inside / P0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(P_times, P_inside / P0, lw=1.5)
    ax1.set_xlabel("t"); ax1.set_ylabel("P_inside / P₀")
    ax1.set_title("Survival probability"); ax1.set_ylim(bottom=0)

    ax2.plot(P_times, T_frac, lw=1.5, color="C1")
    ax2.set_xlabel("t"); ax2.set_ylabel("T(t) = 1 − P/P₀")
    ax2.set_title("Cumulative transmission"); ax2.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, folder, f"transmission{tag}.png")


# ═══════════════════════════════════════════════
# Boundary overlay helper
# ═══════════════════════════════════════════════

def _draw_boundaries(ax, L, chi_R=None, cap_radius=None):
    """
    Draw analysis-mask boundary and CAP onset circle on *ax*.

    chi_R contour works for any geometry (circle, ellipse, stadium).
    """
    n = chi_R.shape[0] if chi_R is not None else 256
    x = np.linspace(-L / 2, L / 2, n)

    if chi_R is not None:
        ax.contour(x, x, chi_R, levels=[0.5],
                   colors=["cyan"], linewidths=[1.2], linestyles=["--"])
    if cap_radius is not None:
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(cap_radius * np.cos(th), cap_radius * np.sin(th),
                color="red", lw=1.0, ls=":", alpha=0.8)


# ═══════════════════════════════════════════════
# Smooth post-propagation playback (FuncAnimation)
# ═══════════════════════════════════════════════

def playback_animation(
    snapshots, snap_times, L,
    chi_R=None, cap_radius=None,
    stride=1, interval_ms=30, cmap="inferno",
    title_fmt="t = {t:.3f}",
    save_path=None, save_fps=30, save_dpi=120,
):
    """
    Smooth replay of pre-recorded snapshots with FuncAnimation.
    Shows mask/CAP boundaries and live transmission in the title.
    """
    idx = np.arange(0, len(snapshots), stride)
    frames = snapshots[idx]
    ftimes = snap_times[idx]
    ext = [-L / 2, L / 2, -L / 2, L / 2]

    # precompute P_inside for each frame
    if chi_R is not None:
        P_arr = np.array([float(np.sum(density(f) * chi_R)) for f in frames])
        P0 = P_arr[0] if P_arr[0] > 0 else 1.0
    else:
        P_arr = None

    fig, ax = plt.subplots(figsize=(6, 5))
    rho0 = density(frames[0]).real
    im = ax.imshow(rho0, extent=ext, origin="lower", cmap=cmap,
                   animated=True)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    _draw_boundaries(ax, L, chi_R=chi_R, cap_radius=cap_radius)

    # text for transmission (bottom-left of axes)
    trans_txt = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                        color="white", fontsize=10, fontweight="bold",
                        va="bottom", ha="left",
                        bbox=dict(facecolor="black", alpha=0.5, pad=2))

    title = ax.set_title(title_fmt.format(t=ftimes[0], step=0))
    fig.tight_layout()

    def _update(i):
        data = density(frames[i]).real
        im.set_array(data)
        title.set_text(title_fmt.format(t=ftimes[i], step=i * stride))
        if P_arr is not None:
            T = 1.0 - P_arr[i] / P0
            trans_txt.set_text(f"T = {T:.4f}")
        return (im, title, trans_txt)

    ani = FuncAnimation(fig, _update, frames=len(frames),
                        interval=interval_ms, blit=False)

    if save_path:
        ext_f = os.path.splitext(save_path)[1].lower()
        if ext_f == ".gif":
            ani.save(save_path, writer="pillow", fps=save_fps, dpi=save_dpi)
        else:
            ani.save(save_path, writer="ffmpeg", fps=save_fps, dpi=save_dpi,
                     extra_args=["-pix_fmt", "yuv420p"])
        print(f"  Animation saved: {save_path}")

    plt.show()
    return ani


# ═══════════════════════════════════════════════
# Real-time live display (during propagation)
# ═══════════════════════════════════════════════

class _DensityDisplay:
    """Base class: shared figure setup, density update, and transmission overlay."""

    def __init__(self, L, chi_R=None, cap_radius=None,
                 cmap="inferno", title_fmt="t = {t:.3f}", figsize=(6, 5)):
        self.title_fmt = title_fmt
        self.chi_R = chi_R
        self.P0 = None
        self._clim_set = False

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ext = [-L / 2, L / 2, -L / 2, L / 2]
        self.im = self.ax.imshow(
            np.zeros((4, 4)), extent=self.ext, origin="lower", cmap=cmap)
        self.fig.colorbar(self.im, ax=self.ax)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        _draw_boundaries(self.ax, L, chi_R=chi_R, cap_radius=cap_radius)

        self.trans_txt = self.ax.text(
            0.02, 0.02, "", transform=self.ax.transAxes,
            color="white", fontsize=10, fontweight="bold",
            va="bottom", ha="left",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))
        self.fig.tight_layout()

    def _update_frame(self, psi, t, step):
        rho = density(psi).real
        self.im.set_array(rho)
        if not self._clim_set:
            self.im.set_clim(vmin=0, vmax=rho.max() or 1e-15)
            self._clim_set = True
        self.ax.set_title(self.title_fmt.format(t=t, step=step))
        if self.chi_R is not None:
            P = float(np.sum(rho * self.chi_R))
            if self.P0 is None:
                self.P0 = P if P > 0 else 1.0
            self.trans_txt.set_text(f"T = {1.0 - P / self.P0:.4f}")


class LiveDisplay(_DensityDisplay):
    """Real-time imshow during propagation (plt.ion)."""

    def __init__(self, L, **kwargs):
        plt.ion()
        super().__init__(L, **kwargs)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, psi, t, step):
        self._update_frame(psi, t, step)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)


class AnimationRecorder(_DensityDisplay):
    """Stream frames to mp4/gif during propagation (file-only)."""

    def __init__(self, filename, L, fps=30, dpi=120, **kwargs):
        kwargs.setdefault("figsize", (5.5, 5))
        kwargs.setdefault("title_fmt", "t = {t:.2f}")
        super().__init__(L, **kwargs)
        self.filename = filename
        self._started = True

        ext = os.path.splitext(filename)[1].lower()
        if ext == ".gif":
            self.writer = PillowWriter(fps=fps)
        else:
            self.writer = FFMpegWriter(
                fps=fps, codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"])
        self.writer.setup(self.fig, filename, dpi=dpi)

    def update(self, psi, t, step):
        self._update_frame(psi, t, step)

    def finish(self):
        if self._started:
            self.writer.finish()
            plt.close(self.fig)
            self._started = False
            print(f"  Animation saved: {self.filename}")
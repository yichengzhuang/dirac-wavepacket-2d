#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D potential functions and analysis masks.

Hard-wall, soft-wall, and harmonic confinement for circle, ellipse,
and Bunimovich stadium geometries.
"""

import numpy as np


# ═══════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════

def _smooth_heaviside(s, smooth_width):
    """Smooth step using tanh: 0 for s≪0, 1 for s≫0."""
    w = max(float(smooth_width), 1e-12)
    return 0.5 * (1.0 + np.tanh(s / w))


def _stadium_form_sq(X, Y, a_half, b):
    """
    Squared form function f²(r) for a Bunimovich stadium.

    Stadium = rectangle |x| ≤ a_half, |y| ≤ b, capped with semicircles
    of radius b at x = ±a_half.

    f = 1 on boundary, < 1 inside, > 1 outside.
    f² = [max(|x| − a_half, 0)² + y²] / b²
    """
    x_excess = np.clip(np.abs(X) - a_half, 0.0, None)
    return (x_excess**2 + Y**2) / b**2


# ═══════════════════════════════════════════════
# Hard wall (tanh step)
# ═══════════════════════════════════════════════

def hard_wall_circle(X, Y, radius=2.0, V0=5e3, smooth_width=0.03):
    R = np.sqrt(X**2 + Y**2)
    return V0 * _smooth_heaviside(R - radius, smooth_width)


def hard_wall_ellipse(X, Y, a=2.0, b=1.2, V0=5e3, smooth_width=0.03):
    rho = np.sqrt((X / a)**2 + (Y / b)**2)
    return V0 * _smooth_heaviside(rho - 1.0, smooth_width)


# ═══════════════════════════════════════════════
# Soft wall  —  V(r) = V0 / (1 + λ exp[μ (1 − f²)])
#
# Inside  (f < 1): V ≈ 0      (flat bottom)
# Outside (f > 1): V ≈ V0     (barrier)
# Boundary (f=1):  V = V0/(1+λ)
#
# λ controls flatness of interior, μ controls wall steepness.
# ═══════════════════════════════════════════════

def soft_wall_circle(X, Y, radius=2.0, V0=70.0, lam=25.0, mu=5.0):
    """Soft-wall circular confinement."""
    f2 = (X**2 + Y**2) / radius**2
    return V0 / (1.0 + lam * np.exp(mu * (1.0 - f2)))


def soft_wall_ellipse(X, Y, a=2.0, b=1.2, V0=70.0, lam=25.0, mu=5.0):
    """Soft-wall elliptical confinement."""
    f2 = (X / a)**2 + (Y / b)**2
    return V0 / (1.0 + lam * np.exp(mu * (1.0 - f2)))


def soft_wall_stadium(X, Y, a_half=2.0, b=1.0, V0=70.0, lam=25.0, mu=5.0):
    """Soft-wall Bunimovich stadium confinement."""
    f2 = _stadium_form_sq(X, Y, a_half, b)
    return V0 / (1.0 + lam * np.exp(mu * (1.0 - f2)))


# ═══════════════════════════════════════════════
# Harmonic (quadratic bowl)
# ═══════════════════════════════════════════════

def harmonic_circle(X, Y, k_harm=3.0, r0=2.0):
    R = np.sqrt(X**2 + Y**2)
    return 0.5 * k_harm * (R / r0)**2


def harmonic_ellipse(X, Y, k_harm=3.0, a=2.0, b=1.2):
    rho2 = (X / a)**2 + (Y / b)**2
    return 0.5 * k_harm * rho2


# ═══════════════════════════════════════════════
# Masks (sharp characteristic functions)
# ═══════════════════════════════════════════════

def circular_mask(X, Y, radius):
    return (np.sqrt(X**2 + Y**2) <= radius).astype(float)


def elliptical_mask(X, Y, aR, bR):
    return ((X / aR)**2 + (Y / bR)**2 <= 1.0).astype(float)


def stadium_mask(X, Y, a_half, b):
    """χ = 1 inside the stadium (f² ≤ 1)."""
    return (_stadium_form_sq(X, Y, a_half, b) <= 1.0).astype(float)